"""
ios-photos-tumblr-sync

Reads photos from a macOS Photos album and posts each photo to Tumblr
backdated to the original EXIF capture date.
"""

import os
import sys
import time
import tempfile
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import osxphotos
import pytumblr
from dotenv import load_dotenv
from PIL import Image
from pillow_heif import register_heif_opener
from timezonefinder import TimezoneFinder

register_heif_opener()  # enables Pillow to open HEIC/HEIF files

LOG_FILE = "posted_photos.log"
DELAY_SECONDS = 360  # 6 minutes — keeps under Tumblr's 250 posts/day limit

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
    if photo.path is None:
        # File is iCloud-only and not downloaded locally — cannot export without Photos.app
        return None

    exported = photo.export(tmpdir, use_photos_export=False, overwrite=True)
    if not exported:
        return None
    exported_path = exported[0]

    # Convert HEIC/HEIF/DNG/WebP to JPEG (pillow-heif handles HEIC decode)
    if not exported_path.lower().endswith((".jpg", ".jpeg")):
        jpeg_path = os.path.splitext(exported_path)[0] + ".jpg"
        with Image.open(exported_path) as img:
            img.convert("RGB").save(jpeg_path, "JPEG", quality=95)
        return jpeg_path

    return exported_path


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


def build_tags() -> list[str]:
    return ["food"]


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
        data=[jpeg_path],  # pytumblr expects a list
    )
    # pytumblr returns either {meta: {status: 201}, response: {id: ...}}
    # or directly {id: ..., state: published} depending on API version
    if "errors" in response:
        raise RuntimeError(f"Tumblr API error: {response}")
    status = response.get("meta", {}).get("status", 0)
    if status not in (0, 200, 201):  # 0 = no meta wrapper (direct response)
        raise RuntimeError(f"Tumblr API error: {response}")
    post_id = (
        response.get("response", {}).get("id")
        or response.get("id")
        or "unknown"
    )
    return str(post_id)


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

    # Verify Tumblr credentials and blog name before processing any photos
    blog_info = tumblr_client.blog_info(config["TUMBLR_BLOG_NAME"])
    if "blog" not in blog_info:
        print(f"ERROR: Could not access Tumblr blog '{config['TUMBLR_BLOG_NAME']}'.")
        print(f"Response: {blog_info}")
        print("Check TUMBLR_BLOG_NAME and your OAuth credentials in .env.")
        sys.exit(1)
    print(f"Tumblr blog verified: {blog_info['blog']['title']}\n")

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
                if photo.path is None:
                    print(f"  SKIP: Photo is iCloud-only (not downloaded to this Mac).")
                else:
                    print(f"  WARNING: Export failed for {photo.uuid}, skipping.")
                continue

            caption = build_caption(metadata)
            tags = build_tags()
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
