API_BASE_URLimport os
import sys
from pathlib import Path
from dotenv import load_dotenv
import requests
from io import BytesIO
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ConversationHandler,
    ContextTypes,
    filters,
)

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
    text = update.message.text or ""
    chat_id = update.effective_chat.id
    username = update.effective_user.username

    # Future existing-user link:
    # /start link_abc123
    print(text,chat_id,username)
    if text.startswith("/start link_"):
        token = text.replace("/start link_", "").strip()
        await handle_link_token(update, token, chat_id, username)
        return

    await update.message.reply_text(
        "👋 Welcome to Family Focus\n\n"
        "To register a new family, send:\n"
        "/registerfamily"
    )

async def handle_photo(update, context):
    message = update.message
    chat_id = message.chat_id

    if not message.photo:
        return

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

    await message.reply_text(text)


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


if __name__ == "__main__":
    main()