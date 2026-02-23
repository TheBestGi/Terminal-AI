import os, sys, time, base64, json, re
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

# --- Setup Project Environment ---
PROJECT_NAME = "TerminalAI_Pro"
BASE_DIR = os.path.expanduser(f"~/{PROJECT_NAME}")
IMAGE_DIR = os.path.join(BASE_DIR, "images")
ENV_FILE = os.path.join(BASE_DIR, ".env")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")

os.makedirs(IMAGE_DIR, exist_ok=True)
console = Console()

def get_config():
    """Handles Token and Environment persistence."""
    token = None
    if os.path.exists(ENV_FILE):
        try:
            with open(ENV_FILE, "r") as f:
                for line in f:
                    if line.startswith("HF_TOKEN="):
                        token = line.split("=")[1].strip()
        except OSError: pass # Prevent crash on read error
    
    if not token:
        console.print(Panel("[bold yellow]🔑 FIRST-TIME SETUP[/bold yellow]\nPlease enter your [bold cyan]Hugging Face Token[/bold cyan]:"))
        token = console.input("[bold green]Token: [/bold green]").strip()
        try:
            with open(ENV_FILE, "w") as f:
                f.write(f"HF_TOKEN={token}")
            console.print(f"[green]✅ Token saved to {ENV_FILE}[/green]")
        except OSError as e:
            console.print(f"[red]Failed to save config: {e}[/red]")
    
    return token

TOKEN = get_config()
client = InferenceClient(api_key=TOKEN)

# --- Persistence Layer ---
def load_mem():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                return (
                    data.get("history", []), 
                    data.get("deep_mem", {"files": {}}), 
                    data.get("custom_role", "You are a local developer AI.")
                )
        except Exception: pass
    return [], {"files": {}}, "You are a local developer AI."

def save_mem(history, deep_mem, role):
    try:
        # Use a temporary file pattern to avoid descriptor errors during writes
        with open(MEMORY_FILE, 'w') as f:
            json.dump({
                "history": history[-20:], 
                "deep_mem": deep_mem, 
                "custom_role": role
            }, f)
    except OSError: pass

chat_history, DEEP_MEMORY, CUSTOM_ROLE = load_mem()

# --- Model Definitions ---
MODELS = {
    "1": ("DeepSeek V3", "deepseek-ai/DeepSeek-V3", "chat"),
    "2": ("DeepSeek R1", "deepseek-ai/DeepSeek-R1", "chat"),
    "3": ("Qwen 2.5 VL", "Qwen/Qwen2.5-VL-7B-Instruct", "vision"),
    "4": ("Flux.1 Schnell", "black-forest-labs/FLUX.1-schnell", "image"),
    "5": ("Qwen 2.5 Coder", "Qwen/Qwen2.5-Coder-32B-Instruct", "chat"),
    "6": ("Llama 3.3 70B", "meta-llama/Llama-3.3-70B-Instruct", "chat")
}

def select_model():
    console.clear()
    console.print(Panel(f"[bold cyan]🛠️ {PROJECT_NAME.upper()} ACTIVE[/bold cyan]\n[dim]Workspace: {BASE_DIR}[/dim]"))
    for k, (n, _, t) in MODELS.items():
        console.print(f"[bold magenta][{k}][/bold magenta] {n} [dim]({t})[/dim]")
    choice = console.input("\nBrain ID: ").strip()
    return MODELS.get(choice, MODELS["1"])

MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model()

# --- Core Processing ---
def handle_file_writing(text):
    pattern = r"SAVE_FILE:\s*([a-zA-Z0-9_\-\.]+)\s*\n(.*?)(?=SAVE_FILE:|END_SAVE|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for filename, content in matches:
        fpath = os.path.join(BASE_DIR, filename.strip())
        clean_content = re.sub(r"```[a-z]*\n", "", content).replace("```", "").strip()
        try:
            with open(fpath, "w") as f: 
                f.write(clean_content)
            console.print(Panel(f"[bold green]💾 AUTO-SAVED:[/bold green] {fpath}", border_style="green"))
        except Exception as e: 
            console.print(f"[red]Write Error: {e}[/red]")

def run_ai(user_text, search_data=None):
    global chat_history
    
    if MODEL_TYPE == "image":
        with console.status("[bold magenta]Fluxing..."):
            try:
                img = client.text_to_image(user_text, model=MODEL_ID)
                path = os.path.join(IMAGE_DIR, f"flux_{int(time.time())}.png")
                img.save(path)
                console.print(f"[green]🎨 Saved: {path}[/green]")
            except Exception as e: 
                console.print(f"[red]Flux Error: {e}[/red]")
            return

    txt_context, img_list = "", []
    for name, data in DEEP_MEMORY["files"].items():
        if str(data).startswith("data:image"):
            img_list.append({"type": "image_url", "image_url": {"url": data}})
        else:
            txt_context += f"FILE ({os.path.basename(name)}):\n{data}\n---\n"
    
    if search_data: 
        txt_context += f"WEB_RESEARCH:\n{search_data}\n---\n"

    sys_prompt = f"{CUSTOM_ROLE}\nProject Path: {BASE_DIR}\nContext:\n{txt_context}\nTo write files, use 'SAVE_FILE: filename.ext' and end with 'END_SAVE'."
    user_payload = f"{sys_prompt}\n\nUSER_QUERY: {user_text}"
    
    content = img_list + [{"type": "text", "text": user_payload}] if (MODEL_TYPE == "vision" and img_list) else user_payload
    msgs = chat_history + [{"role": "user", "content": content}]

    full_resp = ""
    console.print(f"\n[bold magenta]{MODEL_NAME}[/bold magenta]:")
    
    # Live display with safety handling for stream interruption
    with Live(Markdown("💭 *Thinking...*"), console=console, auto_refresh=True) as live:
        try:
            stream = client.chat_completion(model=MODEL_ID, messages=msgs, stream=True, max_tokens=4000)
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, 'reasoning_content', "") or getattr(delta, 'content', "") or ""
                    full_resp += str(token)
                    live.update(Markdown(full_resp.replace("<think>", "💭 *Thinking...*\n").replace("</think>", "\n---\n")))
            
            handle_file_writing(full_resp)
            chat_history.append({"role": "user", "content": user_text})
            chat_history.append({"role": "assistant", "content": full_resp})
            save_mem(chat_history, DEEP_MEMORY, CUSTOM_ROLE)
        except OSError as e:
            if e.errno == 9:
                console.print("[red]\n⚠️ Stream Interrupted (Bad File Descriptor). The connection was closed unexpectedly.[/red]")
            else:
                console.print(f"[red]\nSystem Error: {e}[/red]")
        except Exception as e: 
            console.print(f"[red]\nAI Error: {e}[/red]")

# --- Main CLI ---
while True:
    try: 
        cmd = console.input(f"\n[bold cyan]User[/bold cyan] [dim](search/upload/role/status/wipe/switch/exit)[/dim]: ").strip()
    except (EOFError, KeyboardInterrupt): 
        break
    
    if not cmd or cmd.lower() == "exit": 
        break

    if cmd.lower() == "role":
        CUSTOM_ROLE = console.input("[bold yellow]New System Role: [/bold yellow]").strip()
        save_mem(chat_history, DEEP_MEMORY, CUSTOM_ROLE)
        continue

    if cmd.lower() == "status":
        t = Table(title="AI Context Status")
        t.add_column("Property")
        t.add_column("Value")
        t.add_row("Model", MODEL_NAME)
        t.add_row("Memory", f"{len(DEEP_MEMORY['files'])} objects")
        t.add_row("Token", f"{TOKEN[:8]}****" if TOKEN else "None")
        console.print(t)
        continue

    if cmd.lower() == "switch":
        MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model()
        continue

    if cmd.lower() == "wipe":
        try:
            if os.path.exists(MEMORY_FILE): os.remove(MEMORY_FILE)
            if os.path.exists(ENV_FILE): os.remove(ENV_FILE)
            console.print("[bold red]💥 ALL DATA & CONFIG WIPED.[/bold red]")
            sys.exit(0)
        except OSError as e:
            console.print(f"[red]Wipe failed: {e}[/red]")

    if cmd.lower().startswith("search "):
        search_query = cmd[7:].strip()
        with console.status(f"[bold green]Searching for '{search_query}'..."):
            try:
                # Direct call to DDGS avoids context manager socket reuse issues
                results = DDGS().text(search_query, max_results=5)
                res_text = "\n".join([f"{r.get('href')}: {r.get('body')}" for r in results])
                run_ai(f"Web Research: {search_query}", search_data=res_text)
            except Exception as e:
                console.print(f"[red]Search Error: {e}[/red]")
        continue

    if cmd.lower().startswith("upload"):
        path = cmd[7:].strip().strip("'\"")
        if os.path.exists(path):
            try:
                ext = path.split('.')[-1].lower()
                if ext in ['png', 'jpg', 'jpeg', 'webp']:
                    with open(path, "rb") as f:
                        enc = base64.b64encode(f.read()).decode('utf-8')
                        DEEP_MEMORY["files"][path] = f"data:image/{ext};base64,{enc}"
                else:
                    with open(path, "r", errors="ignore") as f:
                        DEEP_MEMORY["files"][path] = f.read()
                console.print(f"[green]📂 Loaded: {path}[/green]")
            except OSError as e:
                console.print(f"[red]Upload failed: {e}[/red]")
        else:
            console.print(f"[red]File not found: {path}[/red]")
        continue

    run_ai(cmd)

