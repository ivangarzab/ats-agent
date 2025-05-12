"""
Session-related commands (start_emulation)
"""
import discord
from discord import app_commands

from utils.embeds import create_embed

def setup_session_commands(bot):

    @bot.tree.command(name="start_emulation", description="Start the autonomous car emulation")
    async def start_emulation_command(interaction: discord.Interaction):
        """Start the autonomous car emulation through the OpenAI API."""
        await interaction.response.defer()
        
        await bot.openai_service.set_autonomous_car_role()
        
        embed = create_embed(
            title="🤖",
            description="ATSY has driving session has started!",
            color_key="info"
        )
        await interaction.followup.send(embed=embed)
        print("Sent book recommendation command response.")