"""
Helper functions for creating Discord embeds
"""
import discord
from datetime import datetime
import pytz
from utils.constants import COLORS

def create_embed(title, description=None, color_key="info", fields=None, footer=None, timestamp=False):
    """
    Create a standardized Discord embed
    
    Args:
        title (str): The embed title
        description (str, optional): The embed description
        color_key (str, optional): Key for the color in the COLORS dictionary. Defaults to "info".
        fields (list, optional): List of field dicts with name, value, and inline keys
        footer (str, optional): Footer text
        timestamp (bool, optional): Whether to add a timestamp. Defaults to False.
    
    Returns:
        discord.Embed: The created embed
    """
    embed = discord.Embed(
        title=title,
        description=description,
        color=COLORS.get(color_key, COLORS["info"])
    )
    
    # Add fields if provided
    if fields:
        for field in fields:
            embed.add_field(
                name=field["name"],
                value=field["value"],
                inline=field.get("inline", False)
            )
    
    # Add footer if provided
    if footer:
        embed.set_footer(text=footer)
    
    # Add timestamp if requested
    if timestamp:
        sf_timezone = pytz.timezone('US/Pacific')
        embed.timestamp = datetime.now(tz=sf_timezone)
    
    return embed