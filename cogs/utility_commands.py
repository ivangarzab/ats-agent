"""
Utility commands (funfact, robot)
"""
import random
import discord
from discord import app_commands

from utils.constants import FUN_FACTS, FACT_CLOSERS
from utils.embeds import create_embed

def setup_utility_commands(bot):
    """
    Setup utility commands for the bot
    
    Args:
        bot: The bot instance
    """

    @bot.tree.command(name="funfact", description="Get a random book-related fun fact")
    async def funfact_command(interaction: discord.Interaction):
        embed = create_embed(
            title="📚 Book Fun Fact",
            description=random.choice(FUN_FACTS),
            color_key="purp",
            footer=random.choice(FACT_CLOSERS)
        )
        await interaction.response.send_message(embed=embed)
        print("Sent funfact command response.")
        
    @bot.tree.command(name="robot", description="Ask me something (uses AI)")
    @app_commands.describe(prompt="What do you want to ask?")
    async def robot_command(interaction: discord.Interaction, prompt: str):
        """Make prompt to OpenAI."""
        await interaction.response.defer()  # Defer the response since API call might take time
        
        response = await bot.openai_service.get_response(prompt)
        embed = create_embed(
            title="🤖 Robot Response",
            description=response,
            color_key="blank"
        )
        await interaction.followup.send(embed=embed)
        print("Sent robot command response.")
        
    # Also register the text-based robot command
    @bot.command()
    async def robot(ctx, *, prompt: str):
        """Make prompt to OpenAI."""
        response = await bot.openai_service.get_response(prompt)
        embed = create_embed(
            title="🤖 Robot Response",
            description=response,
            color_key="blank"
        )
        await ctx.send(embed=embed)