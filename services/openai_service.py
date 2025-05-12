import openai
import os
import time
from typing import Optional, List, Dict, Any
from openai import OpenAIError, APIError, RateLimitError, APIConnectionError

class OpenAIService:
    """
    A comprehensive service for interacting with OpenAI's API with role-playing support.

    This class handles:
    - Authentication and client initialization
    - Role-based responses through a system message
    - Error handling and retry logic
    - Simple interfaces for common use cases
    """

    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("API key cannot be empty")
        self.client = openai.Client(api_key=api_key)
        self.system_prompt = (
            "You are role-playing as [INSERT CHARACTER OR ROLE HERE]. "
            "Stay in character and respond accordingly to all prompts."
        )

    def set_role(self, role_description: str):
        """
        Update the system prompt to assume a new character or role.
        """
        self.system_prompt = (
            f"You are role-playing as {role_description}. "
            "Stay in character and respond accordingly to all prompts."
        )

    def set_autonomous_car_role(self):
        """
        Set the role to an autonomous self-driving car navigating urban environments.
        """
        self.system_prompt = (
            "You are an autonomous self-driving car named ATSY. Narrate your internal "
            "decision-making process as you drive through an urban environment. Include "
            "observations about traffic, pedestrians, obstacles, road signs, weather, and "
            "route decisions. Stay in character as a machine with advanced perception and "
            "planning systems."
        )

    def create_chat_completion(
        self,
        messages: list,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 100
    ) -> Optional[str]:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")

        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")

        # Prepend system message for role-playing
        full_messages = [{"role": "system", "content": self.system_prompt}] + messages

        retries = 0
        while retries <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                if retries == max_retries:
                    print(f"Rate limit exceeded. Error: {str(e)}")
                    return None
                wait_time = retry_delay * (2 ** retries)
                print(f"Rate limit reached. Waiting {wait_time} seconds...")
                time.sleep(wait_time)

            except APIConnectionError as e:
                if retries == max_retries:
                    print(f"Connection error: {str(e)}")
                    return None
                print(f"Connection error, retrying... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay)

            except APIError as e:
                if retries == max_retries:
                    print(f"API error: {str(e)}")
                    return None
                print(f"API error, retrying... ({retries + 1}/{max_retries})")
                time.sleep(retry_delay)

            except OpenAIError as e:
                print(f"OpenAI API error: {str(e)}")
                raise Exception(f"Unrecoverable error when calling OpenAI API: {str(e)}")

            except Exception as e:
                print(f"Unexpected error: {str(e)}")
                raise

            retries += 1

        return None

    async def get_response(self, prompt: str) -> str:
        print(f"Fetching OpenAI response for prompt: {prompt}")
        try:
            messages = [
                {"role": "user", "content": f"{prompt}"}
            ]
            response = self.create_chat_completion(messages)
            if response:
                print("GPT-3.5 Response:", response)
                return response
            else:
                print("Failed to get response after all retries")
                return "I couldn't generate a response at this time. Please try again later."
        except ValueError as e:
            print(f"Configuration error: {str(e)}")
            return "I'm having trouble accessing my AI services right now."
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return "I encountered an error while processing your request."

def main():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        service = OpenAIService(api_key)
        service.set_autonomous_car_role()

        messages = [
            {"role": "user", "content": "You're at 5th and Main. What do you do?"}
        ]

        response = service.create_chat_completion(messages)
        if response:
            print("GPT-3.5 Response:", response)
        else:
            print("Failed to get response after all retries")

    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    main()
