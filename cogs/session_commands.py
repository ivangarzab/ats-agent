"""
Session-related commands (book, duedate, session, discussions)
"""
import discord
from discord import app_commands

from utils.embeds import create_embed

def setup_session_commands(bot):
    """
    Setup session-related commands for the bot
    
    Args:
        bot: The bot instance
    """
    async def _get_active_session(interaction):
        """
        Helper function to get active session data
        
        Args:
            interaction: The Discord interaction
            
        Returns:
            Tuple of (club_data, session_data) if successful
            If no active session, sends a message and returns None, None
        """
        # Get the club ID from bot config (or use a default if not available)
        club_id = getattr(bot.config, 'DEFAULT_CLUB_ID', 'club-1')
        
        # Get club data from API
        club_data = bot.api.get_club(club_id)
        
        # Check if there's an active session
        if not club_data.get('active_session'):
            await interaction.followup.send("There is no active reading session right now.")
            return None, None
            
        return club_data, club_data['active_session']

    @bot.tree.command(name="book", description="Show current book details")
    async def book_command(interaction: discord.Interaction):
        await interaction.response.defer()  # Defer the response while we fetch data
        
        # Get active session data
        club_data, session = await _get_active_session(interaction)
        if not session:
            return
            
        book = session['book']
        
        embed = create_embed(
            title="📚 Current Book",
            description=f"**{book['title']}**",
            color_key="info",
            fields=[
                {"name": "Author", "value": f"{book['author']}"}
            ],
            footer="Happy reading! 📖"
        )
        
        # Add extra book details if available
        if book.get('year'):
            embed.add_field(name="Year", value=str(book['year']), inline=True)
        if book.get('edition'):
            embed.add_field(name="Edition", value=book['edition'], inline=True)
            
        await interaction.followup.send(embed=embed)
        print("Sent book command response.")

    @bot.tree.command(name="duedate", description="Show the session's due date")
    async def duedate_command(interaction: discord.Interaction):
        await interaction.response.defer()
        
        # Get active session data
        club_data, session = await _get_active_session(interaction)
        if not session:
            return
            
        due_date = session['due_date']
        
        embed = create_embed(
            title="📅 Due Date",
            description=f"Session due date: **{due_date}**",
            color_key="warning"
        )
        await interaction.followup.send(embed=embed)
        print("Sent duedate command response.")

    @bot.tree.command(name="session", description="Show current session details")
    async def session_command(interaction: discord.Interaction):
        await interaction.response.defer()
        
        # Get active session data
        club_data, session = await _get_active_session(interaction)
        if not session:
            return
            
        book = session['book']
        
        fields = [
            {
                "name": "Book",
                "value": f"{book['title']}",
                "inline": True
            },
            {
                "name": "Author",
                "value": f"{book['author']}",
                "inline": True
            },
            {
                "name": "Due Date",
                "value": f"{session['due_date']}",
                "inline": False
            }
        ]
        
        # Add discussion count if available
        if session.get('discussions') and len(session['discussions']) > 0:
            fields.append({
                "name": "Discussions",
                "value": f"{len(session['discussions'])} scheduled",
                "inline": True
            })
        
        embed = create_embed(
            title="📚 Current Session Details",
            color_key="info",
            fields=fields,
            footer="Keep reading! 📖"
        )
        await interaction.followup.send(embed=embed)
        print("Sent session command response.")

    @bot.tree.command(name="discussions", description="Show the session's discussion details")
    async def discussions_command(interaction: discord.Interaction):
        await interaction.response.defer()
        
        # Get active session data
        club_data, session = await _get_active_session(interaction)
        if not session:
            return
            
        # Check if there are any discussions
        if not session.get('discussions') or len(session['discussions']) == 0:
            await interaction.followup.send("There are no discussions scheduled for this session.")
            return
            
        discussions = session['discussions']
        
        # Sort discussions by date
        discussions.sort(key=lambda x: x['date'])
        
        # Create fields for each discussion
        fields = []
        for i, discussion in enumerate(discussions):
            fields.append({
                "name": f"Discussion {i+1}: {discussion['title']}",
                "value": f"**Date**: {discussion['date']}\n**Location**: {discussion.get('location', 'TBD')}",
                "inline": False
            })
        
        embed = create_embed(
            title="📚 Book Discussion Details",
            color_key="info",
            fields=fields,
            footer="Don't stop reading! 📖"
        )
        await interaction.followup.send(embed=embed)
        print("Sent discussions command response.")
    
    @bot.tree.command(name="book_summary", description="Let me provide a summary of the active book")
    async def booksummary_command(interaction: discord.Interaction):
        """Ask OpenAI for a summary of the active book."""
        await interaction.response.defer()
        
        # Get active session data
        club_data, session = await _get_active_session(interaction)
        if not session:
            return
            
        book_title = session['book']['title']
        
        response = await bot.openai_service.get_response(
            f"What is {book_title} about?"
        )
        embed = create_embed(
            title="🤖 Book Summary",
            description=response,
            color_key="info"
        )
        await interaction.followup.send(embed=embed)
        print("Sent book summary command response.")