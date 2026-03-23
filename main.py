"""
ios-photos-tumblr-sync

Reads photos from a local folder and posts each to Tumblr backdated to the
original EXIF capture date.

If ANTHROPIC_API_KEY is set in .env, Claude Vision is used to detect food
items and add specific tags (e.g. #pizza, #sushi). If not set, posts are
tagged with #food only.
"""

import base64
import os
import sys
import time
from datetime import datetime, timezone
from io import BytesIO
from zoneinfo import ZoneInfo

import pytumblr
from dotenv import load_dotenv
from PIL import Image, ExifTags
from pillow_heif import register_heif_opener
from timezonefinder import TimezoneFinder

try:
    import anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

register_heif_opener()  # enables Pillow to open HEIC/HEIF files

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "posted_photos.log")
DELAY_SECONDS = 360  # 6 minutes — stays under Tumblr's 250 posts/day limit
SUPPORTED_EXT = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp", ".dng"}
CLAUDE_MODEL = "claude-haiku-4-5-20251001"
MAX_IMAGE_PX = 1568  # Claude Vision recommended max dimension

_tf = TimezoneFinder()

# EXIF tag IDs
_TAG_IDS = {v: k for k, v in ExifTags.TAGS.items()}
TAG_DATE_ORIGINAL = _TAG_IDS.get("DateTimeOriginal", 36867)
TAG_GPS_INFO      = _TAG_IDS.get("GPSInfo", 34853)
TAG_MAKE          = _TAG_IDS.get("Make", 271)
TAG_MODEL         = _TAG_IDS.get("Model", 272)


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_environment() -> dict:
    load_dotenv()
    required = [
        "TUMBLR_CONSUMER_KEY",
        "TUMBLR_CONSUMER_SECRET",
        "TUMBLR_OAUTH_TOKEN",
        "TUMBLR_OAUTH_SECRET",
        "TUMBLR_BLOG_NAME",
        "PHOTOS_FOLDER",
    ]
    config = {}
    missing = []
    for key in required:
        value = os.getenv(key)
        if not value:
            missing.append(key)
        else:
            config[key] = value
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        sys.exit(1)
    # Optional: Anthropic API key for Claude Vision food tag detection
    config["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY") or None
    return config


# ---------------------------------------------------------------------------
# Posted log  (tracks by filename)
# ---------------------------------------------------------------------------

def load_posted_log(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_to_log(log_path: str, filename: str) -> None:
    with open(log_path, "a") as f:
        f.write(filename + "\n")


# ---------------------------------------------------------------------------
# Photo discovery
# ---------------------------------------------------------------------------

def get_photos(folder: str) -> list[str]:
    if not os.path.isdir(folder):
        print(f"ERROR: PHOTOS_FOLDER '{folder}' does not exist.")
        sys.exit(1)
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXT
    ]
    if not files:
        print(f"ERROR: No supported image files found in '{folder}'.")
        sys.exit(1)
    # Sort by EXIF date; fall back to file modification time
    files.sort(key=lambda p: _sort_key(p))
    return files


def _sort_key(path: str):
    meta = extract_metadata(path)
    dt = meta["date"]
    if dt is None:
        return datetime.fromtimestamp(os.path.getmtime(path))
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


# ---------------------------------------------------------------------------
# EXIF extraction
# ---------------------------------------------------------------------------

def extract_metadata(path: str) -> dict:
    exif_raw = {}
    try:
        with Image.open(path) as img:
            exif_raw = img._getexif() or {}
    except Exception:
        pass

    date = _parse_exif_date(exif_raw.get(TAG_DATE_ORIGINAL))
    lat, lon = _parse_gps(exif_raw.get(TAG_GPS_INFO))
    make  = (exif_raw.get(TAG_MAKE)  or "").strip()
    model = (exif_raw.get(TAG_MODEL) or "").strip()

    if make and model.startswith(make):
        camera = model
    elif make and model:
        camera = f"{make} {model}"
    elif model:
        camera = model
    else:
        camera = None

    return {
        "date":     date,
        "latitude": lat,
        "longitude": lon,
        "camera":   camera,
        "filename": os.path.basename(path),
    }


def _parse_exif_date(raw) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.strptime(str(raw), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None


def _parse_gps(gps_info) -> tuple:
    if not gps_info:
        return None, None
    try:
        def to_deg(val):
            return float(val[0]) + float(val[1]) / 60 + float(val[2]) / 3600
        lat = to_deg(gps_info[2])
        lon = to_deg(gps_info[4])
        if gps_info.get(1) == "S":
            lat = -lat
        if gps_info.get(3) == "W":
            lon = -lon
        return lat, lon
    except Exception:
        return None, None


# ---------------------------------------------------------------------------
# Claude Vision food tag detection (optional)
# ---------------------------------------------------------------------------

def _encode_image_for_claude(path: str) -> tuple[str, str]:
    with Image.open(path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_PX:
            ratio = MAX_IMAGE_PX / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
    return data, "image/jpeg"


def detect_food_tags(client, path: str) -> list[str]:
    """
    Use Claude Vision to identify food items in the photo.
    Returns a list of single-word lowercase tags (e.g. ['pizza', 'salad']).
    Returns an empty list if no food is detected or on any error.
    """
    try:
        data, media_type = _encode_image_for_claude(path)
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=256,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": data,
                            },
                        },
                        {
                            "type": "text",
                            "text": (
                                "List the food items visible in this photo as a "
                                "comma-separated list of single-word lowercase tags "
                                "(e.g., pizza, sushi, ramen). If no food is visible, "
                                "respond with exactly: none. Only output the tag list, "
                                "nothing else."
                            ),
                        },
                    ],
                }
            ],
        )
        raw = response.content[0].text.strip().lower()
        if raw == "none" or not raw:
            return []
        tags = [t.strip().lstrip("#") for t in raw.split(",") if t.strip()]
        return [t for t in tags if t and t != "none"]
    except Exception as e:
        print(f"  WARNING: Claude Vision failed ({type(e).__name__}: {e}), skipping food tags.")
        return []


# ---------------------------------------------------------------------------
# Caption, tags, date
# ---------------------------------------------------------------------------

def build_caption(metadata: dict) -> str:
    parts = []
    dt = metadata["date"]
    if dt:
        parts.append(dt.strftime("%B %d, %Y"))
    if metadata.get("camera"):
        parts.append(f"Shot on {metadata['camera']}")
    return " | ".join(parts) if parts else ""


def build_tags(food_tags: list[str] | None = None) -> list[str]:
    return ["food"] + (food_tags or [])


def format_tumblr_date(metadata: dict) -> str:
    dt = metadata["date"]
    if dt is None:
        # Fall back to now if no EXIF date
        return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    lat, lon = metadata.get("latitude"), metadata.get("longitude")

    if lat is not None and lon is not None:
        tz_name = _tf.timezone_at(lat=lat, lng=lon)
        local_tz = ZoneInfo(tz_name) if tz_name else datetime.now().astimezone().tzinfo
    else:
        local_tz = datetime.now().astimezone().tzinfo

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)

    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tumblr
# ---------------------------------------------------------------------------

def post_to_tumblr(client, blog_name, jpeg_path, caption, tags, date_str) -> str:
    response = client.create_photo(
        blog_name,
        state="published",
        tags=tags,
        caption=caption,
        date=date_str,
        data=[jpeg_path],
    )
    if "errors" in response:
        raise RuntimeError(f"Tumblr API error: {response}")
    status = response.get("meta", {}).get("status", 0)
    if status not in (0, 200, 201):
        raise RuntimeError(f"Tumblr API error: {response}")
    return str(response.get("response", {}).get("id") or response.get("id") or "unknown")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_environment()
    folder = config["PHOTOS_FOLDER"]

    posted = load_posted_log(LOG_FILE)
    photos = get_photos(folder)

    total = len(photos)
    already_posted = sum(1 for p in photos if os.path.basename(p) in posted)
    remaining = total - already_posted
    print(f"Found {total} photo(s) in '{folder}'.")
    print(f"{already_posted} already posted, {remaining} to go.\n")

    if remaining == 0:
        print("Nothing to post. All photos have already been synced.")
        return

    tumblr_client = pytumblr.TumblrRestClient(
        config["TUMBLR_CONSUMER_KEY"],
        config["TUMBLR_CONSUMER_SECRET"],
        config["TUMBLR_OAUTH_TOKEN"],
        config["TUMBLR_OAUTH_SECRET"],
    )

    blog_info = tumblr_client.blog_info(config["TUMBLR_BLOG_NAME"])
    if "blog" not in blog_info:
        print(f"ERROR: Could not access Tumblr blog '{config['TUMBLR_BLOG_NAME']}'.")
        print(f"Response: {blog_info}")
        sys.exit(1)
    print(f"Tumblr blog verified: {blog_info['blog']['title']}\n")

    # Set up Claude Vision client if API key is available
    anthropic_client = None
    if config["ANTHROPIC_API_KEY"] and _ANTHROPIC_AVAILABLE:
        import anthropic as _anthropic
        anthropic_client = _anthropic.Anthropic(api_key=config["ANTHROPIC_API_KEY"])
        print("Claude Vision enabled: food tags will be AI-detected.\n")
    else:
        print("Claude Vision disabled: posts will be tagged with #food only.")
        print("Set ANTHROPIC_API_KEY in .env to enable AI food tag detection.\n")

    posts_this_run = 0
    for i, photo_path in enumerate(photos):
        filename = os.path.basename(photo_path)
        label = f"[{i + 1}/{total}]"

        if filename in posted:
            print(f"{label} Skipping {filename} (already posted)")
            continue

        print(f"{label} Processing {filename}...")
        metadata = extract_metadata(photo_path)

        food_tags = detect_food_tags(anthropic_client, photo_path) if anthropic_client else []
        caption  = build_caption(metadata)
        tags     = build_tags(food_tags)
        date_str = format_tumblr_date(metadata)

        print(f"  Date:    {date_str} (UTC)")
        print(f"  Caption: {caption}")
        print(f"  Tags:    {', '.join(f'#{t}' for t in tags)}")

        try:
            post_id = post_to_tumblr(
                tumblr_client,
                config["TUMBLR_BLOG_NAME"],
                photo_path,
                caption,
                tags,
                date_str,
            )
            save_to_log(LOG_FILE, filename)
            posted.add(filename)
            posts_this_run += 1
            print(f"  Posted successfully. Post ID: {post_id}")
        except Exception as e:
            print(f"  ERROR posting {filename}: {e}")
            continue

        remaining_after = sum(1 for p in photos if os.path.basename(p) not in posted)
        if remaining_after > 0:
            print(f"  Waiting {DELAY_SECONDS // 60} minutes before next post...")
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Posted {posts_this_run} new photo(s) this run.")


if __name__ == "__main__":
    main()
