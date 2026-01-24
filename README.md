
# Terminal AI: Ultimate Edition

A professional-grade CLI developer environment. Bridge the gap between local files, live web research, high-end reasoning models, and AI image generation.

##  Quick Start

1. **Clone the Directory:**
   ```bash
   git clone https://github.com/TheBestGi/Terminal-AI
   cd terminal-ai

2: **Install required dependencies.**

Make sure you have a modern version of Python3 installed. 

In your terminal type "pip3 install huggingface_hub" and then "pip3 install rich" and then "pip3 install duckduckgo-search". This installs required system packages.
    

🔑 **First Boot Setup:**
On your initial launch, the system will detect a missing configuration and guide you through the Setup Wizard:
API Token: You will be prompted to enter your Hugging Face API Token.

Get a Token: If you don't have one, generate it for free at huggingface.co/settings/tokens.

Security: Your token is saved locally in a .env file and is never uploaded or shared.

⌨️ **Command Console**:

Enter these keywords at the User prompt to trigger specific system tools:

**Command Actions**：

upload <path>: Loads a text file or Image (PNG/JPG) into Deep Memory.

search <query>: Triggers DDGS Web Research to fetch live data from the internet.

status: Shows the active model, chat history length, and a table of loaded files.

forget: Opens a menu to selectively remove specific files/images from memory.

switch: Instantly swap between DeepSeek, Llama Vision, and Flux models.

wipe: Nukes all conversation history and all loaded context files.

exit: Securely synchronizes memory to memory.json and shuts down.

💾 **AI File-Writing Protocol**

This system allows the AI to write code or data directly to your local workspace. To trigger an auto-save, the AI (specifically Qwen Coder) is trained to use this exact syntax:

SAVE_FILE: script.py
print("This file was written by AI")
END_SAVE


The terminal will detect this block, strip the tags, and save script.py to your current directory automatically.

🧠 **Model Lineup**

DeepSeek V3 (685B): General purpose high-end intelligence for complex queries.

DeepSeek R1: Specialized for deep logic, math, and Chain-of-Thought reasoning.

Llama 3.2 Vision: Optimized for image analysis. Use after uploading a photo.

Flux.1 Schnell: High-speed, high-quality image generation (saves to /images).

Qwen 2.5 Coder: Optimized for Python/Web development and local file writing.

📁 **Project Structure**

terminal_ai.py: The main execution engine.

.env: Stores your API credentials (created on first boot).

memory.json: Stores your persistent chat history and deep memory.

/images: Directory where Flux-generated art is saved.

# Advanced Users

1: Feel free to add your own models and if you have a paid plan you can use those models as well. If you don't know what you are doing then stick with the provided models.

2: Please report bugs to me immediately. 

3: Thanks for using this software!
