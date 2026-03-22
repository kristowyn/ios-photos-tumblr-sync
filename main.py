"""
ios-photos-tumblr-sync

Reads photos from a macOS Photos album, identifies food items using Claude Vision,
and posts each photo to Tumblr backdated to the original EXIF capture date.
"""

import base64
import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from io import BytesIO
from zoneinfo import ZoneInfo

import anthropic
import osxphotos
import pytumblr
from dotenv import load_dotenv
from PIL import Image
from timezonefinder import TimezoneFinder

LOG_FILE = "posted_photos.log"
DELAY_SECONDS = 360  # 6 minutes — keeps under Tumblr's 250 posts/day limit
MAX_IMAGE_PX = 1568  # Claude Vision optimal max dimension
CLAUDE_MODEL = "claude-3-5-haiku-20241022"

_tf = TimezoneFinder()


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def load_environment() -> dict:
    load_dotenv()
    keys = [
        "TUMBLR_CONSUMER_KEY",
        "TUMBLR_CONSUMER_SECRET",
        "TUMBLR_OAUTH_TOKEN",
        "TUMBLR_OAUTH_SECRET",
        "TUMBLR_BLOG_NAME",
        "ANTHROPIC_API_KEY",
        "PHOTOS_ALBUM_NAME",
    ]
    config = {}
    missing = []
    for key in keys:
        value = os.getenv(key)
        if not value:
            missing.append(key)
        else:
            config[key] = value
    if missing:
        print(f"ERROR: Missing required environment variables: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in all values.")
        sys.exit(1)
    return config


# ---------------------------------------------------------------------------
# Posted log
# ---------------------------------------------------------------------------

def load_posted_log(log_path: str) -> set:
    if not os.path.exists(log_path):
        return set()
    with open(log_path, "r") as f:
        return {line.strip() for line in f if line.strip()}


def save_to_log(log_path: str, uuid: str) -> None:
    with open(log_path, "a") as f:
        f.write(uuid + "\n")


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------

def get_album_photos(album_name: str) -> list:
    print(f"Opening Photos library...")
    db = osxphotos.PhotosDB()
    photos = db.photos(albums=[album_name])
    photos = [p for p in photos if p.isphoto]
    if not photos:
        print(f"ERROR: No photos found in album '{album_name}'.")
        print("Check that PHOTOS_ALBUM_NAME in .env matches an existing album exactly.")
        sys.exit(1)
    photos.sort(key=lambda p: p.date)
    return photos


def extract_metadata(photo) -> dict:
    exif = photo.exif_info
    place = photo.place

    camera_parts = []
    if exif:
        if exif.camera_make:
            camera_parts.append(exif.camera_make)
        if exif.camera_model:
            camera_parts.append(exif.camera_model)
    camera_str = " ".join(camera_parts) if camera_parts else None

    location_str = None
    if place:
        location_str = place.name or None

    return {
        "uuid": photo.uuid,
        "date": photo.date,
        "latitude": photo.latitude,
        "longitude": photo.longitude,
        "location_str": location_str,
        "camera": camera_str,
        "filename": photo.original_filename,
    }


def export_as_jpeg(photo, tmpdir: str) -> str | None:
    exported = photo.export(
        tmpdir,
        use_photos_export=False,
        convert_to_jpeg=True,
        overwrite=True,
    )
    if not exported:
        return None
    return exported[0]


# ---------------------------------------------------------------------------
# Claude Vision
# ---------------------------------------------------------------------------

def encode_image_for_claude(jpeg_path: str) -> tuple[str, str]:
    with Image.open(jpeg_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_IMAGE_PX:
            ratio = MAX_IMAGE_PX / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        data = base64.b64encode(buf.getvalue()).decode()
    return data, "image/jpeg"


def detect_food_tags(client: anthropic.Anthropic, jpeg_path: str) -> list[str]:
    try:
        data, media_type = encode_image_for_claude(jpeg_path)
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
        print(f"  WARNING: Claude Vision failed ({e}), continuing without food tags.")
        return []


# ---------------------------------------------------------------------------
# Caption and tags
# ---------------------------------------------------------------------------

def build_caption(metadata: dict) -> str:
    parts = []

    date_obj = metadata["date"]
    parts.append(date_obj.strftime("%B %d, %Y"))

    if metadata.get("location_str"):
        parts.append(metadata["location_str"])

    if metadata.get("camera"):
        parts.append(f"Shot on {metadata['camera']}")

    return " | ".join(parts)


def build_tags(food_tags: list[str]) -> list[str]:
    return ["food"] + food_tags


# ---------------------------------------------------------------------------
# Date / timezone
# ---------------------------------------------------------------------------

def format_tumblr_date(metadata: dict) -> str:
    dt = metadata["date"]
    lat = metadata.get("latitude")
    lon = metadata.get("longitude")

    # Determine the timezone where the photo was taken
    if lat is not None and lon is not None:
        tz_name = _tf.timezone_at(lat=lat, lng=lon)
        if tz_name:
            local_tz = ZoneInfo(tz_name)
        else:
            local_tz = datetime.now().astimezone().tzinfo
    else:
        local_tz = datetime.now().astimezone().tzinfo

    # If datetime is naive (osxphotos sometimes returns naive datetimes),
    # attach the determined local timezone
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)
    else:
        dt = dt.astimezone(local_tz)

    # Convert to UTC for Tumblr
    dt_utc = dt.astimezone(timezone.utc)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------------------
# Tumblr
# ---------------------------------------------------------------------------

def post_to_tumblr(
    client,
    blog_name: str,
    jpeg_path: str,
    caption: str,
    tags: list[str],
    date_str: str,
) -> str:
    response = client.create_photo(
        blog_name,
        state="published",
        tags=tags,
        caption=caption,
        date=date_str,
        data=jpeg_path,
    )
    status = response.get("meta", {}).get("status", 0)
    if status not in (200, 201):
        raise RuntimeError(f"Tumblr API error: {response}")
    return str(response.get("response", {}).get("id", "unknown"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = load_environment()

    posted = load_posted_log(LOG_FILE)
    photos = get_album_photos(config["PHOTOS_ALBUM_NAME"])

    total = len(photos)
    already_posted = sum(1 for p in photos if p.uuid in posted)
    remaining = total - already_posted
    print(f"Found {total} photo(s) in album '{config['PHOTOS_ALBUM_NAME']}'.")
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
    anthropic_client = anthropic.Anthropic(api_key=config["ANTHROPIC_API_KEY"])

    posts_this_run = 0
    for i, photo in enumerate(photos):
        label = f"[{i + 1}/{total}]"

        if photo.uuid in posted:
            print(f"{label} Skipping {photo.original_filename} (already posted)")
            continue

        print(f"{label} Processing {photo.original_filename}...")
        metadata = extract_metadata(photo)

        with tempfile.TemporaryDirectory() as tmpdir:
            jpeg_path = export_as_jpeg(photo, tmpdir)
            if not jpeg_path:
                print(f"  WARNING: Export failed for {photo.uuid}, skipping.")
                continue

            food_tags = detect_food_tags(anthropic_client, jpeg_path)
            caption = build_caption(metadata)
            tags = build_tags(food_tags)
            date_str = format_tumblr_date(metadata)

            print(f"  Date:    {date_str} (UTC)")
            print(f"  Caption: {caption}")
            print(f"  Tags:    {', '.join(f'#{t}' for t in tags)}")

            try:
                post_id = post_to_tumblr(
                    tumblr_client,
                    config["TUMBLR_BLOG_NAME"],
                    jpeg_path,
                    caption,
                    tags,
                    date_str,
                )
                save_to_log(LOG_FILE, photo.uuid)
                posted.add(photo.uuid)
                posts_this_run += 1
                print(f"  Posted successfully. Post ID: {post_id}")
            except Exception as e:
                print(f"  ERROR posting {photo.uuid}: {e}")
                continue

        # Wait between posts, but not after the last one
        remaining_after = sum(1 for p in photos if p.uuid not in posted)
        if remaining_after > 0:
            print(f"  Waiting {DELAY_SECONDS // 60} minutes before next post...")
            time.sleep(DELAY_SECONDS)

    print(f"\nDone. Posted {posts_this_run} new photo(s) this run.")


if __name__ == "__main__":
    main()
