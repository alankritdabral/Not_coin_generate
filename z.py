from telethon import TelegramClient

# Your API ID and Hash from my.telegram.org
api_id = '27129612'
api_hash = '6718a94ee7173552f55abb9c7edd4819'

# Your phone number associated with your Telegram account
phone_number = '+919105567268'

# Create the client and connect
client = TelegramClient('session_name', api_id, api_hash)

async def main():
    # Connect and sign in
    await client.start(phone=phone_number)

    # Send message to the bot
    await client.send_message('@Free_Notcoin_Instant_Pay_Bot', '🎁 Claim Not 💰')

    print("Message sent!")

# Run the client
with client:
    client.loop.run_until_complete(main())
