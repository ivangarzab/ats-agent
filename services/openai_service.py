import openai
import os
import time
from typing import Optional, List, Dict, Any
from openai import OpenAIError, APIError, RateLimitError, APIConnectionError

class AttentionSchema:
    """
    Python implementation of Attention Schema Theory for an autonomous vehicle.
    Represents the car's model of its own attention mechanism.
    """
    def __init__(self):
        self.current_focus = None  # Current primary input being attended to
        self.attention_queue = []  # Priority queue of inputs
        self.attention_intensity = 1.0  # The current "strength" of attention
        self.attention_properties = {
            "is_voluntary": True,  # Whether attention is deliberately directed
            "is_stable": True,     # Whether attention is fluctuating
            "effort": 0.5          # How much effort is being exerted
        }
        self.attention_history = []  # Recent attention shifts (limited to 5 most recent)
        self.max_history = 5
    
    def add_to_history(self, input_obj):
        """Add an input to attention history and maintain maximum size"""
        self.attention_history.insert(0, input_obj)
        if len(self.attention_history) > self.max_history:
            self.attention_history.pop()
    
    def to_dict(self):
        """Convert to dictionary representation for response formatting"""
        return {
            "currentFocus": self.current_focus.to_dict() if self.current_focus else None,
            "attentionQueue": [input_obj.to_dict() for input_obj in self.attention_queue[:3]],  # Top 3 items
            "attentionIntensity": self.attention_intensity,
            "attentionProperties": self.attention_properties,
            "attentionHistory": [input_obj.to_dict() for input_obj in self.attention_history]
        }

class Input:
    """Represents an input that the autonomous car can attend to"""
    def __init__(self, source, content, priority=0.0, category=None):
        self.source = source          # e.g., "visual", "radar", "lidar", "map"
        self.content = content        # The actual information
        self.priority = priority      # Higher = more important
        self.category = category      # e.g., "obstacle", "traffic sign", "pedestrian"
        self.timestamp = time.time()
    
    def to_dict(self):
        """Convert to dictionary representation for response formatting"""
        return {
            "source": self.source,
            "content": self.content,
            "priority": self.priority,
            "category": self.category
        }

class AttentionState:
    """Represents the current state of the attention system"""
    def __init__(self, schema, processing_capacity=1.0, processing_load=0.0):
        self.current_schema = schema
        self.processing_capacity = processing_capacity  # Available resources (0.0-1.0)
        self.processing_load = processing_load          # Current demand (0.0-1.0)
    
    def to_dict(self):
        """Convert to dictionary representation for response formatting"""
        return {
            "currentSchema": self.current_schema.to_dict(),
            "processingCapacity": self.processing_capacity,
            "processingLoad": self.processing_load
        }

class OpenAIService:
    """
    A comprehensive service for interacting with OpenAI's API with role-playing support.
    
    This class handles:
    - Authentication and client initialization
    - Role-based responses through a system message with Attention Schema Theory implementation
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
        # Initialize the attention schema
        self.attention_schema = AttentionSchema()
        self.attention_state = AttentionState(self.attention_schema)
    
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
        Set the role to an autonomous self-driving car with Attention Schema Theory implementation.
        """
        self.system_prompt = (
            "You are an autonomous self-driving car named ATSY with an implemented Attention Schema. "
            "When responding, format your output as a JSON representation of your Attention Schema, "
            "showing your current focus, attention queue of prioritized inputs, attention properties, "
            "and attention history. Then provide a brief natural language summary of your current state. "
            "Your responses should model how attention works in an autonomous system, showing how you "
            "prioritize different sensory inputs and navigational decisions."
        )

    def update_attention_schema(self, user_input: str):
        """
        Update the attention schema based on the user's input
        """
        # Process the user input to determine what the car should be focusing on
        # This is a simplified version that creates appropriate inputs based on keywords
        
        # Reset processing load to base level
        self.attention_state.processing_load = 0.4
        
        # Create potential inputs based on the user's question
        potential_inputs = []
        
        # Check for different types of queries and create appropriate inputs
        if "looking" in user_input.lower() or "see" in user_input.lower():
            potential_inputs.append(Input("visual", "Road ahead with traffic flow", 0.9, "navigation"))
            potential_inputs.append(Input("visual", "Traffic signals at intersection", 0.8, "regulation"))
            potential_inputs.append(Input("visual", "Pedestrians on sidewalk", 0.7, "safety"))
            self.attention_state.processing_load = 0.7  # Visual processing is demanding
            
        if "location" in user_input.lower() or "where" in user_input.lower():
            potential_inputs.append(Input("gps", "Current location coordinates", 0.85, "navigation"))
            potential_inputs.append(Input("map", "Route planning information", 0.75, "navigation"))
            
        if "traffic" in user_input.lower():
            potential_inputs.append(Input("lidar", "Vehicle density analysis", 0.8, "traffic"))
            potential_inputs.append(Input("visual", "Traffic flow patterns", 0.75, "traffic"))
            self.attention_state.processing_load = 0.65
            
        if "weather" in user_input.lower():
            potential_inputs.append(Input("sensors", "Current weather conditions", 0.6, "environment"))
            potential_inputs.append(Input("visual", "Road surface conditions", 0.7, "safety"))
            
        if "obstacle" in user_input.lower() or "hazard" in user_input.lower():
            potential_inputs.append(Input("lidar", "Obstacle detection data", 0.95, "safety"))
            potential_inputs.append(Input("radar", "Moving object tracking", 0.9, "safety"))
            self.attention_state.processing_load = 0.85  # Safety critical processing
            
        # If no specific inputs were created, add some default ones
        if not potential_inputs:
            potential_inputs.append(Input("visual", "General road observation", 0.7, "navigation"))
            potential_inputs.append(Input("sensors", "System status check", 0.5, "maintenance"))
            potential_inputs.append(Input("map", "Navigation progress", 0.6, "navigation"))
        
        # Sort by priority
        potential_inputs.sort(key=lambda x: x.priority, reverse=True)
        
        # Update the attention schema
        self.attention_schema.attention_queue = potential_inputs
        
        # Set the current focus to the highest priority input
        old_focus = self.attention_schema.current_focus
        self.attention_schema.current_focus = potential_inputs[0] if potential_inputs else None
        
        # If focus changed, add to history
        if old_focus != self.attention_schema.current_focus and self.attention_schema.current_focus:
            self.attention_schema.add_to_history(self.attention_schema.current_focus)
        
        # Adjust attention properties based on the situation
        if self.attention_state.processing_load > 0.8:
            self.attention_schema.attention_properties["effort"] = 0.9
            self.attention_schema.attention_properties["is_stable"] = False
        elif self.attention_state.processing_load > 0.6:
            self.attention_schema.attention_properties["effort"] = 0.7
            self.attention_schema.attention_properties["is_stable"] = True
        else:
            self.attention_schema.attention_properties["effort"] = 0.5
            self.attention_schema.attention_properties["is_stable"] = True
            
        # Update the attention intensity based on the priority of the current focus
        if self.attention_schema.current_focus:
            self.attention_schema.attention_intensity = min(self.attention_schema.current_focus.priority * 1.2, 1.0)
        else:
            self.attention_schema.attention_intensity = 0.5
    
    def create_chat_completion(
        self,
        messages: list,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        max_tokens: int = 500  # Increased token limit for detailed responses
    ) -> Optional[str]:
        if not messages:
            raise ValueError("Messages list cannot be empty")

        if not isinstance(messages, list):
            raise ValueError("Messages must be a list of dictionaries")

        for message in messages:
            if not isinstance(message, dict) or 'role' not in message or 'content' not in message:
                raise ValueError("Each message must be a dictionary with 'role' and 'content' keys")
        
        # Update attention schema based on the last user message
        last_user_message = next((m["content"] for m in reversed(messages) if m["role"] == "user"), None)
        if last_user_message:
            self.update_attention_schema(last_user_message)
        
        # Get the attention state as a dictionary
        attention_state_dict = self.attention_state.to_dict()
        
        # Add attention state info to the system prompt
        attention_prompt = (
            f"{self.system_prompt}\n\n"
            f"Your attention state is currently:\n"
            f"Current focus: {attention_state_dict['currentSchema']['currentFocus']}\n"
            f"Processing capacity: {attention_state_dict['processingCapacity']}\n"
            f"Processing load: {attention_state_dict['processingLoad']}\n"
            f"Format your response as a JSON structure reflecting your attention schema, "
            f"followed by a natural language explanation."
        )

        # Prepend system message for role-playing
        full_messages = [{"role": "system", "content": attention_prompt}] + messages

        retries = 0
        while retries <= max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=full_messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                raw_response = response.choices[0].message.content
                
                # Format the response with attention schema details
                import json
                
                # Create a more compact JSON representation without the deeper nesting
                compact_json = {
                    "focus": {
                        "source": self.attention_schema.current_focus.source if self.attention_schema.current_focus else None,
                        "content": self.attention_schema.current_focus.content if self.attention_schema.current_focus else None,
                        "priority": self.attention_schema.current_focus.priority if self.attention_schema.current_focus else None,
                        "category": self.attention_schema.current_focus.category if self.attention_schema.current_focus else None
                    },
                    "attentionQueue": [{"source": i.source, "content": i.content, "priority": i.priority} 
                                     for i in self.attention_schema.attention_queue[:2]],
                    "attentionProperties": self.attention_schema.attention_properties,
                    "processing": {
                        "capacity": self.attention_state.processing_capacity,
                        "load": self.attention_state.processing_load
                    }
                }
                
                attention_json = json.dumps(compact_json, indent=2)
                
                # Ensure the JSON and natural language response are clearly separated
                formatted_response = f"```json\n{attention_json}\n```\n\n{raw_response}"
                slef.logger.info(f"[LOG] {raw_response}")
                return formatted_response

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

def demo_car_attention(api_key: str):
    """
    Demonstrate the autonomous car with attention schema theory implementation
    """
    service = OpenAIService(api_key)
    service.set_autonomous_car_role()
    
    # Define some sample prompts to demonstrate the attention schema in action
    sample_prompts = [
        "What are you currently looking at?",
        "Is there any traffic ahead?",
        "Do you see any obstacles in your path?",
        "What's your current location?",
        "How are the weather conditions affecting your driving?"
    ]
    
    print("\n=== AUTONOMOUS CAR ATTENTION SCHEMA DEMO ===\n")
    
    for prompt in sample_prompts:
        print(f"\n>>> USER: {prompt}\n")
        response = service.create_chat_completion([{"role": "user", "content": prompt}])
        if response:
            print(f">>> ATSY (Autonomous Vehicle):\n{response}\n")
            print("-" * 80)
        else:
            print("Failed to get response after all retries")
    
    print("\nDemo completed.")

def main():
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        # Run the attention schema demo
        demo_car_attention(api_key)

    except ValueError as e:
        print(f"Configuration error: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()