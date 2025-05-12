"""
Configuration module for the bot
"""
import os
from dotenv import load_dotenv

class BotConfig:
    """Configuration class to handle environment variables and settings"""
    def __init__(self):
        load_dotenv(override=True)
        
        # Channel configuration
        self.DEFAULT_CHANNEL = 1327357851827572872
        
        # Environment detection
        self.ENV = os.getenv("ENV")
        if self.ENV == "dev":
            print("[DEBUG] ~~~~~~~~~~~~ Running in development mode ~~~~~~~~~~~~")
            self.TOKEN = os.getenv("TOKEN")
        else:
            self.TOKEN = os.getenv("TOKEN")
        
        # API Keys    
        self.KEY_OPENAI = os.getenv("KEY_OPEN_AI")
        
        # Print debug information
        self._debug_print()
        
        # Validate configuration
        self._validate()
    
    def _debug_print(self):
        """Print debug information about configuration"""
        print(f"[DEBUG] TOKEN: {'SET' if self.TOKEN else 'NOT SET'}")
        print(f"[DEBUG] KEY_OPENAI: {'SET' if self.KEY_OPENAI else 'NOT SET'}")
    
    def _validate(self):
        """Validate that required configuration is present"""
        if not self.TOKEN:
            raise ValueError("[ERROR] TOKEN environment variable is not set.")