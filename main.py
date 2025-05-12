#!/usr/bin/env python3
"""
Main entry point for the ATS-Agent bot.
"""
import os
from bot import ATSAgent

def main():
    """Main function to start the bot"""
    bot = ATSAgent()
    bot.run(bot.config.TOKEN)

if __name__ == "__main__":
    main()