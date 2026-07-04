import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from io import BytesIO
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

import asyncio
from collections import defaultdict

media_groups = defaultdict(list)
media_group_tasks = {}

# Allow importing db_helpers/app helpers from project root
ROOT_DIR = Path(__file__).resolve().parent.parent
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:5000")
BOT_API_SECRET = os.getenv("BOT_API_SECRET")


sys.path.append(str(ROOT_DIR))

from db_helpers import database_read, database_write  # adjust if your names differ

load_dotenv(ROOT_DIR / ".env")

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

ASK_FAMILY_NAME, ASK_EMAIL = range(2)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)

    # Existing web user linking flow:
    # /start link_xxxxx
    if context.args:
        start_arg = context.args[0]

        if start_arg.startswith("link_"):
            token = start_arg.replace("link_", "", 1)

            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/telegram/link-account",
                    json={
                        "token": token,
                        "telegram_chat_id": chat_id,
                        "telegram_username": update.effective_user.username,
                    },
                    timeout=20,
                )

                data = response.json()

                if response.status_code == 200 and data.get("success"):
                    await update.message.reply_text(
                        "✅ Telegram connected successfully!\n\n"
                        "You can now send photos here and Family Focus will detect your family members."
                    )
                    return

                await update.message.reply_text(
                    "❌ Could not connect Telegram.\n\n"
                    f"{data.get('error', 'The link may be expired or already used.')}"
                )
                return

            except Exception as e:
                print("Telegram link failed:", e)
                await update.message.reply_text(
                    "❌ Could not connect Telegram right now. Please try again."
                )
                return

    # Normal /start
    status_data = get_telegram_connection_status(chat_id)

    if status_data and status_data.get("linked"):
        family_name = status_data.get("family_name", "your family")

        await update.message.reply_text(
            f"👋 Welcome back to Family Focus!\n\n"
            f"Connected family: {family_name}\n\n"
            "📷 Send me photos and I’ll detect your family members."
        )
        return

    await update.message.reply_text(
        "👋 Welcome to Family Focus!\n\n"
        "This Telegram account is not connected yet.\n\n"
        "If you are a new user:\n"
        "➡️ Use /registerfamily\n\n"
        "If you already have a Family Focus account:\n"
        "➡️ Connect Telegram from your Family Focus dashboard."
    )

async def handle_photo(update, context):
    message = update.message

    if not message or not message.photo:
        return

    chat_id = str(update.effective_chat.id)

    status_data = get_telegram_connection_status(chat_id)

    if not status_data or not status_data.get("linked"):
        await message.reply_text(
            "This Telegram account is not connected yet.\n\n"
            "New user? Use /registerfamily\n\n"
            "Already have an account? Connect Telegram from the Family Focus dashboard."
        )
        return

    media_group_id = message.media_group_id

    # Telegram album / bulk upload
    if media_group_id:
        media_groups[media_group_id].append(update)

        if media_group_id not in media_group_tasks:
            media_group_tasks[media_group_id] = asyncio.create_task(
                process_media_group_later(media_group_id, context)
            )

        return

    # Single photo upload
    await process_single_photo(update, context)

async def process_media_group_later(media_group_id, context):
    await asyncio.sleep(3)
    album_url = None
    updates = media_groups.pop(media_group_id, [])
    media_group_tasks.pop(media_group_id, None)

    if not updates:
        return

    first_message = updates[0].message
    chat_id = first_message.chat_id

    await first_message.reply_text(f"📸 {len(updates)} photos received. Scanning...")

    total_confirmed_photos = 0
    total_possible_photos = 0
    total_skipped_photos = 0

    confirmed_names = []
    possible_names = []

    for update in updates:
        message = update.message
        photo = message.photo[-1]

        telegram_file = await context.bot.get_file(photo.file_id)

        file_bytes = BytesIO()
        await telegram_file.download_to_memory(out=file_bytes)
        file_bytes.seek(0)

        files = {
            "photo": ("telegram_photo.jpg", file_bytes, "image/jpeg")
        }

        data = {
            "telegram_chat_id": str(chat_id),
            "telegram_file_id": photo.file_id,
            "telegram_message_id": str(message.message_id),
            "caption": message.caption or ""
        }

        try:
            response = requests.post(
                f"{API_BASE_URL}/api/telegram/upload-photo",
                data=data,
                files=files,
                timeout=120
            )

            result = response.json()
            if result.get("album_url"):
                album_url = result.get("album_url")
        except Exception as e:
            total_skipped_photos += 1
            continue

        if not result.get("ok"):
            total_skipped_photos += 1
            continue

        scan = result.get("result", {})

        if scan.get("confirmed_count", 0) > 0:
            total_confirmed_photos += 1

            for m in scan.get("matches", []):
                name = m.get("name")
                distance = m.get("distance")

                print("TELEGRAM CONFIRMED MATCH:", name, distance)

                if name:
                    confirmed_names.append(name)

        elif scan.get("possible_count", 0) > 0:
            total_possible_photos += 1

            for m in scan.get("possible_matches", []):
                name = m.get("name")
                distance = m.get("distance")

                print("TELEGRAM POSSIBLE MATCH:", name, distance)

                if name:
                    possible_names.append(name)

        else:
            total_skipped_photos += 1

    confirmed_names = sorted(set(confirmed_names))
    possible_names = sorted(set(possible_names))

    text = "✅ Bulk scan complete\n\n"
    text += f"🟢 Confirmed photos: {total_confirmed_photos}\n"
    text += f"🟡 Possible photos: {total_possible_photos}\n"
    text += f"⚪ Skipped photos: {total_skipped_photos}\n"

    if confirmed_names:
        text += f"\nConfirmed: {', '.join(confirmed_names)}"

    if possible_names:
        text += f"\nPossible: {', '.join(possible_names)}"

    reply_markup = None

    if album_url:
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("📷 View Family Album", url=album_url)]
        ])

    await first_message.reply_text(
        text,
        reply_markup=reply_markup
    )

async def register_family(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("What is your family name?")
    return ASK_FAMILY_NAME

async def ask_email(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["family_name"] = update.message.text.strip()
    await update.message.reply_text("What email should be connected to this family?")
    return ASK_EMAIL

async def save_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    email = update.message.text.strip()
    family_name = context.user_data.get("family_name")

    chat_id = update.effective_chat.id
    username = update.effective_user.username

    database_write(
        """
        INSERT INTO telegram_registration_requests
        (family_name, email, telegram_chat_id, telegram_username, status, created_at)
        VALUES (?, ?, ?, ?, 'pending', datetime('now'))
        """,
        (family_name, email, str(chat_id), username),
    )

    await update.message.reply_text(
        "✅ Registration request received.\n\n"
        f"Family: {family_name}\n"
        f"Email: {email}\n\n"
        "Your request is waiting for admin approval."
    )

    return ConversationHandler.END

async def handle_link_token(update: Update, token: str, chat_id: int, username: str | None):
    try:
        print("CALLING FLASK LINK API")
        print("TOKEN:", token)
        print("CHAT ID:", chat_id)
        print("USERNAME:", username)
        print("URL:", f"{API_BASE_URL}/api/telegram/link-account")

        response = requests.post(
            f"{API_BASE_URL}/api/telegram/link-account",
            json={
                "token": token,
                "telegram_chat_id": chat_id,
                "telegram_username": username
            },
            timeout=15
        )

        print("STATUS CODE:", response.status_code)
        print("RAW RESPONSE:", response.text)

        data = response.json()
        print("JSON RESPONSE:", data)

    except Exception as e:
        print("Telegram link error:", e)
        await update.message.reply_text(
            "❌ Could not connect to Family Focus right now.\n"
            "Please try again in a few minutes."
        )
        return

    if data.get("success"):
        await update.message.reply_text(
            "✅ Telegram connected to Family Focus!\n\n"
            "You can now send family photos here, and I'll add them to your Family Focus account."
        )
    else:
        await update.message.reply_text(
            f"❌ {data.get('message', 'Could not connect Telegram.')}"
        )

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Registration cancelled.")
    return ConversationHandler.END

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = str(update.effective_chat.id)

    try:
        response = requests.post(
            API_BASE_URL + "/api/telegram/status",
            json={"telegram_chat_id": chat_id},
            timeout=10,
        )

        if response.status_code != 200:
            await update.message.reply_text(
                f"Could not check registration status right now. Error: {response.status_code}"
            )
            return

        data = response.json()

        if not data.get("found"):
            await update.message.reply_text(
                "You do not have a registration request yet.\n\n"
                "To register, send:\n/registerfamily"
            )
            return

        await update.message.reply_text(
            "📌 Registration Status\n\n"
            f"Family: {data['family_name']}\n"
            f"Email: {data['email']}\n"
            f"Status: {data['status']}"
        )

    except Exception as e:
        await update.message.reply_text(
            f"Could not check registration status right now.\n\n{e}"
        )

async def process_single_photo(update, context):
    message = update.message
    chat_id = message.chat_id

    await message.reply_text("📸 Photo received. Scanning...")

    photo = message.photo[-1]
    telegram_file = await context.bot.get_file(photo.file_id)

    file_bytes = BytesIO()
    await telegram_file.download_to_memory(out=file_bytes)
    file_bytes.seek(0)

    files = {
        "photo": ("telegram_photo.jpg", file_bytes, "image/jpeg")
    }

    data = {
        "telegram_chat_id": str(chat_id),
        "telegram_file_id": photo.file_id,
        "telegram_message_id": str(message.message_id),
        "caption": message.caption or ""
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/api/telegram/upload-photo",
            data=data,
            files=files,
            timeout=120
        )

        result = response.json()

    except Exception as e:
        await message.reply_text(f"❌ Upload failed: {e}")
        return

    if not result.get("ok"):
        await message.reply_text(
            f"❌ Scan failed: {result.get('error', 'Unknown error')}"
        )
        return

    scan = result.get("result", {})
    album_url = result.get("album_url")

    reply_markup = None
    if album_url:
        reply_markup = InlineKeyboardMarkup([
            [InlineKeyboardButton("📷 View Family Album", url=album_url)]
        ])

    await message.reply_text(
        build_scan_message(scan),
        reply_markup=reply_markup
    )



def main():
    if not BOT_TOKEN:
        raise RuntimeError("Missing TELEGRAM_BOT_TOKEN in .env")

    app = ApplicationBuilder().token(BOT_TOKEN).build()

    conv = ConversationHandler(
        entry_points=[CommandHandler("registerfamily", register_family)],
        states={
            ASK_FAMILY_NAME: [MessageHandler(filters.TEXT & ~filters.COMMAND, ask_email)],
            ASK_EMAIL: [MessageHandler(filters.TEXT & ~filters.COMMAND, save_request)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )

    app.add_handler(CommandHandler("start", start))
    app.add_handler(conv)
    app.add_handler(CommandHandler("status", status))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Family Focus Telegram Bot is running...")
    app.run_polling()

#Helper function to build the scan message
def build_scan_message(scan):
    confirmed_names = scan.get("confirmed_names", [])
    possible_names = scan.get("possible_names", [])

    confirmed_count = scan.get("confirmed_count", len(confirmed_names))
    possible_count = scan.get("possible_count", len(possible_names))
    skipped_count = scan.get("skipped_count", 0)

    text = "✅ Scan complete\n\n"

    if confirmed_count:
        text += f"🟢 Confirmed: {', '.join(confirmed_names)}\n"

    if possible_count:
        text += f"🟡 Possible: {', '.join(possible_names)}\n"

    if skipped_count:
        text += f"⚪ Skipped faces: {skipped_count}\n"

    if not confirmed_count and not possible_count and not skipped_count:
        text += "No known family members detected."
    
    if scan.get("status") == "already_rejected":
        rejected_names = scan.get("rejected_names", [])

        return (
            "📸 Scan complete.\n\n"
            "This photo matched someone you already rejected before:\n\n"
            f"🚫 Already rejected: {', '.join(rejected_names)}\n\n"
            "So it was not added back to Review."
        )
    return text

def get_telegram_connection_status(chat_id):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/telegram/connection-status",
            json={"telegram_chat_id": str(chat_id)},
            timeout=20
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:
        print("Telegram connection status check failed:", e)
        return None


if __name__ == "__main__":
    main()