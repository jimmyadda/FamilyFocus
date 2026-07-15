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
    CallbackQueryHandler,
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

                print("TELEGRAM LINK URL:", response.url)
                print("TELEGRAM LINK STATUS:", response.status_code)
                print("TELEGRAM LINK CONTENT-TYPE:", response.headers.get("Content-Type"))
                print("TELEGRAM LINK BODY:", repr(response.text[:1000]))
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
    info = get_start_info(chat_id)

    if not info or not info.get("linked"):
        await update.message.reply_text(
            "👋 Welcome to Family Focus!\n\n"
            "New family?\n"
            "➡️ /registerfamily\n\n"
            "Already have an account?\n"
            "Connect Telegram from your Family Focus dashboard."
        )
        return

    keyboard = []

    if info.get("member_count", 0) > 0:
        keyboard.append([
            InlineKeyboardButton(
                "👤 Upload Member Photos",
                callback_data="upload_member_photos"
            )
        ])

    keyboard.append([
        InlineKeyboardButton(
            "📷 Detect Family Members",
            callback_data="scan_photos"
        )
    ])

    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        f"👋 Welcome back!\n\n"
        f"Connected family: {info['family_name']}\n\n"
        "What would you like to do?",
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
    review_url = result.get("review_url")
    unknown_faces = scan.get("unknown_faces", 0) or 0

    keyboard = []

    if album_url:
        keyboard.append([
            InlineKeyboardButton("📷 View Family Album", url=album_url)
        ])

    if unknown_faces > 0 and review_url:
        keyboard.append([
            InlineKeyboardButton("👀 Review Unknown Faces", url=review_url)
        ])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None

    await message.reply_text(
        build_scan_message(scan),
        reply_markup=reply_markup
    )

async def profile_photo_start(update, context):
    query = update.callback_query
    chat_id = update.effective_chat.id

    try:
        res = requests.get(
            f"{API_BASE_URL}/api/telegram/family-members",
            params={"telegram_chat_id": chat_id},
            timeout=20
        )
        data = res.json()
    except Exception as e:
        print("profile_photo_start error:", e)
        data = {"success": False}

    if not data.get("success"):
        msg = (
            "❌ I could not load your family members.\n\n"
            "Please make sure your Telegram account is connected to Family Focus."
        )

        if query:
            await query.message.reply_text(msg)
        else:
            await update.message.reply_text(msg)
        return

    members = data.get("members", [])

    if not members:
        msg = "No family members found yet. Please add members in Family Focus first."

        if query:
            await query.message.reply_text(msg)
        else:
            await update.message.reply_text(msg)
        return

    keyboard = []

    for member in members:
        keyboard.append([
            InlineKeyboardButton(
                member["name"],
                callback_data=f"profile_member_{member['id']}"
            )
        ])

    keyboard.append([
        InlineKeyboardButton("Cancel", callback_data="profile_cancel")
    ])

    text = "👤 Choose who you want to improve:"

    if query:
        await query.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text(
            text,
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    if query.data == "scan_photos":
        context.user_data["mode"] = "scan"

        await query.message.reply_text(
            "📷 Send me one or more photos and I'll scan them."
        )

        return

    if query.data == "upload_member_photos":
        await profile_photo_start(update, context)
        return
    if query.data.startswith("profile_member_"):
        await profile_member_selected(update, context)
        return

    if query.data == "profile_done":
        await profile_upload_done(update, context)
        return

    if query.data == "profile_cancel":
        await profile_upload_cancel(update, context)
        return
    await query.message.reply_text(
        "👤 Upload Member Photos\n\n"
        "This feature is coming next."
    )
    return

async def handle_photo(update, context):
    message = update.message

    if not message or not message.photo:
        return

    chat_id = str(update.effective_chat.id)
    mode = context.user_data.get("mode", "scan")

    if mode == "profile_upload":
        await process_member_profile_photo(update, context)
        return

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
    
    album_url = None
    review_url = None
    total_confirmed_photos = 0
    total_possible_photos = 0
    total_skipped_photos = 0
    total_unknown_faces = 0

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
            if result.get("review_url"):
                review_url = result.get("review_url")                
        except Exception as e:
            total_skipped_photos += 1
            continue

        if not result.get("ok"):
            total_skipped_photos += 1
            continue

        scan = result.get("result", {})
        
        unknown_faces = scan.get("unknown_faces", 0) or scan.get("skipped_count", 0) or 0
        total_unknown_faces += unknown_faces
        
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
    text += f"👀 Unknown faces: {total_unknown_faces}\n"

    if confirmed_names:
        text += f"\nConfirmed: {', '.join(confirmed_names)}"

    if possible_names:
        text += f"\nPossible: {', '.join(possible_names)}"
    
    keyboard = []
    reply_markup = None

    if album_url:
        keyboard.append([
            InlineKeyboardButton("📷 View Family Album", url=album_url)
        ])

    if total_unknown_faces > 0 and review_url:
        keyboard.append([
            InlineKeyboardButton("👀 Review Unknown Faces", url=review_url)
        ])

    reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
    
    await first_message.reply_text(
        text,
        reply_markup=reply_markup
    )

async def profile_member_selected(update, context):
    query = update.callback_query
    await query.answer()

    member_id = int(query.data.replace("profile_member_", ""))
    member_name = query.message.reply_markup.inline_keyboard

    selected_name = None

    for row in member_name:
        for button in row:
            if button.callback_data == query.data:
                selected_name = button.text
                break

    if not selected_name:
        selected_name = "this member"

    context.user_data["mode"] = "profile_upload"
    context.user_data["member_id"] = member_id
    context.user_data["member_name"] = selected_name
    context.user_data["photos_uploaded"] = 0

    keyboard = [
        [
            InlineKeyboardButton("✅ Done", callback_data="profile_done"),
            InlineKeyboardButton("❌ Cancel", callback_data="profile_cancel")
        ]
    ]

    await query.message.reply_text(
        f"Great!\n\n"
        f"Now send clear photos of {selected_name}.\n\n"
        f"✔ Front\n"
        f"✔ Left\n"
        f"✔ Right\n"
        f"✔ Different lighting\n"
        f"✔ Different expressions\n\n"
        f"Send as many as you'd like.\n\n"
        f"Press Done when finished.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )

async def profile_upload_done(update, context):
    query = update.callback_query
    await query.answer()

    member_name = context.user_data.get("member_name", "member")
    photos_uploaded = context.user_data.get("photos_uploaded", 0)

    context.user_data.clear()

    await query.message.reply_text(
        f"✅ Done.\n\n"
        f"Added {photos_uploaded} profile photos for {member_name}.\n\n"
        f"You are back in normal detection mode."
    )

async def profile_upload_cancel(update, context):
    query = update.callback_query
    await query.answer()

    context.user_data.clear()

    await query.message.reply_text(
        "❌ Profile upload cancelled.\n\n"
        "You are back in normal detection mode."
    )

async def process_member_profile_photo(update, context):
    chat_id = update.effective_chat.id

    member_id = context.user_data.get("member_id")
    member_name = context.user_data.get("member_name", "member")

    if not member_id:
        context.user_data.clear()
        await update.message.reply_text(
            "Something went wrong. Please start profile upload again."
        )
        return

    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)

    file_bytes = await file.download_as_bytearray()

    files = {
        "photo": ("telegram_profile_photo.jpg", bytes(file_bytes), "image/jpeg")
    }

    data = {
        "telegram_chat_id": chat_id,
        "member_id": member_id
    }

    try:
        res = requests.post(
            f"{API_BASE_URL}/api/telegram/upload-member-photo",
            data=data,
            files=files,
            timeout=60
        )
        result = res.json()
    except Exception as e:
        print("handle_profile_photo_upload error:", e)
        result = {"success": False}

    if not result.get("ok"):
        await update.message.reply_text(
            f"❌ Could not add photo to {member_name}.\n\n"
            f"{result.get('error') or result.get('message') or 'Please try another clear face photo.'}"
        )
        return

    context.user_data["photos_uploaded"] = context.user_data.get("photos_uploaded", 0) + 1
    count = context.user_data["photos_uploaded"]

    keyboard = [
        [
            InlineKeyboardButton("✅ Done", callback_data="profile_done"),
            InlineKeyboardButton("❌ Cancel", callback_data="profile_cancel")
        ]
    ]

    await update.message.reply_text(
        f"✅ Photo added to {member_name}\n\n"
        f"Photos uploaded: {count}\n\n"
        f"Send another photo or press Done.",
        reply_markup=InlineKeyboardMarkup(keyboard)
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
    app.add_handler(CallbackQueryHandler(handle_callback))  


    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    print("Family Focus Telegram Bot is running...")
    app.run_polling()

#Helper function to build the scan message

def build_scan_message(scan):
    confirmed_names = scan.get("confirmed_names", [])
    possible_names = scan.get("possible_names", [])

    confirmed_count = scan.get("confirmed_count", len(confirmed_names))
    possible_count = scan.get("possible_count", len(possible_names))
    unknown_faces = scan.get("unknown_faces", 0) or scan.get("skipped_count", 0)

    text = "✅ Scan complete\n\n"

    if confirmed_count:
        text += f"🟢 Confirmed: {', '.join(confirmed_names)}\n"

    if possible_count:
        text += f"🟡 Possible: {', '.join(possible_names)}\n"

    if unknown_faces > 0:
        text += f"👀 Unknown faces: {unknown_faces}\n"
        text += "\nOpen Review to identify them."

    if not confirmed_count and not possible_count and not unknown_faces:
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

def get_start_info(chat_id):
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/telegram/start-info",
            json={"telegram_chat_id": str(chat_id)},
            timeout=20
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception as e:
        print(e)
        return None



if __name__ == "__main__":
    main() 