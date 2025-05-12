"""
Fun commands (rolldice, flipcoin, choose)
"""
import random
import discord
from discord import app_commands

from utils.embeds import create_embed

def setup_fun_commands(bot):
    """
    Setup fun commands for the bot
    
    Args:
        bot: The bot instance
    """
    @bot.tree.command(name="rolldice", description="I will roll a six-sided die")
    async def rolldice_command(interaction: discord.Interaction):
        result = random.randint(1, 6)
        embed = create_embed(
            title="🎲 Dice Roll",
            description=f"You rolled a **{result}**!",
            color_key="fun"
        )
        await interaction.response.send_message(embed=embed)
        print("Sent rolldice command response.")

    @bot.tree.command(name="flipcoin", description="Flip a coin")
    async def flipcoin_command(interaction: discord.Interaction):
        result = random.choice(["HEADS", "TAILS"])
        embed = create_embed(
            title="🪙 Coin Flip",
            description=f"You got **{result}**!",
            color_key="fun"
        )
        await interaction.response.send_message(embed=embed)
        print("Sent flipcoin command response.")

    @bot.tree.command(name="choose", description="I will choose from the options you give me")
    @app_commands.describe(options="Space-separated options to choose from")
    async def choose_command(interaction: discord.Interaction, options: str):
        choices = options.split()
        if not choices:
            await interaction.response.send_message("Please provide some options to choose from!", ephemeral=True)
            return
            
        result = random.choice(choices)
        responses = [
            f"**{result}**, I choose you!",
            f"I have selected **{result}**.",
            f"**{result}**. I have spoken.",
            f"The winner is: **{result}**!"
        ]
        
        embed = create_embed(
            title="🎯 Choice Made",
            description=random.choice(responses),
            color_key="fun"
        )
        await interaction.response.send_message(embed=embed)
        print("Sent choose command response.")