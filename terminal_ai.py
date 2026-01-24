import os, sys, time, base64, json, re
from huggingface_hub import InferenceClient
from duckduckgo_search import DDGS
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.table import Table

# --- Workspace Setup ---
console = Console()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MEMORY_FILE = os.path.join(BASE_DIR, "memory.json")
ENV_FILE = os.path.join(BASE_DIR, ".env")
os.makedirs(IMAGE_DIR, exist_ok=True)

# --- Setup Wizard ---
def get_token():
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as f:
            for line in f:
                if line.startswith("HF_TOKEN="):
                    return line.split("=")[1].strip()
    
    console.print(Panel("[bold yellow]🚀 INITIAL BOOT DETECTED[/bold yellow]\nPlease enter your Hugging Face API Token (hf.co/settings/tokens)"))
    token = console.input("[bold cyan]Token: [/bold cyan]").strip()
    with open(ENV_FILE, 'w') as f: f.write(f"HF_TOKEN={token}")
    return token

TOKEN = get_token()
client = InferenceClient(api_key=TOKEN)

MODELS = {
    "1": ("DeepSeek V3 (685B)", "deepseek-ai/DeepSeek-V3", "chat"),
    "2": ("DeepSeek R1 (Reasoning)", "deepseek-ai/DeepSeek-R1", "chat"),
    "3": ("Flux.1 Schnell (Art)", "black-forest-labs/FLUX.1-schnell", "image"),
    "4": ("Qwen 2.5 Coder 32B", "Qwen/Qwen2.5-Coder-32B-Instruct", "chat")
}

# --- Logic Layer ---
def load_mem():
    if os.path.exists(MEMORY_FILE):
        try:
            with open(MEMORY_FILE, 'r') as f:
                data = json.load(f)
                return data.get("history", []), data.get("deep_mem", {"files": {}})
        except: return [], {"files": {}}
    return [], {"files": {}}

def save_mem(history, deep_mem):
    try:
        with open(MEMORY_FILE, 'w') as f:
            json.dump({"history": history[-10:], "deep_mem": deep_mem}, f)
    except: pass

def select_model():
    console.clear()
    console.print(Panel("[bold cyan]🛠️ TERMINAL AI: ULTIMATE EDITION[/bold cyan]"))
    for k, (n, _, t) in MODELS.items():
        console.print(f"[bold magenta][{k}][/bold magenta] {n} [dim]({t})[/dim]")
    choice = console.input("\nBrain ID: ").strip()
    return MODELS.get(choice, MODELS["1"])

def web_search(query):
    with console.status(f"[bold blue]Searching DDGS..."):
        try:
            with DDGS() as ddgs:
                results = [r for r in ddgs.text(query, max_results=5)]
                return "\n".join([f"Source: {r['href']}\nSnippet: {r['body']}\n" for r in results])
        except Exception as e: return f"Search Error: {e}"

def handle_file_writing(text):
    pattern = r"SAVE_FILE:\s*([a-zA-Z0-9_\-\.]+)\s*\n(.*?)(?=SAVE_FILE:|END_SAVE|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for filename, content in matches:
        fpath = os.path.join(BASE_DIR, filename.strip())
        clean = re.sub(r"```[a-z]*\n", "", content).replace("```", "").strip()
        if clean:
            try:
                with open(fpath, "w") as f: f.write(clean)
                console.print(Panel(f"[bold green]💾 AUTO-SAVED:[/bold green] {fpath}", border_style="green"))
            except Exception as e: console.print(f"[red]Write Error: {e}[/red]")

# --- Inference ---
def run_ai(user_text, search_data=None):
    global chat_history
    if MODEL_TYPE == "image":
        with console.status("[bold magenta]Fluxing..."):
            try:
                img = client.text_to_image(user_text, model=MODEL_ID)
                path = os.path.join(IMAGE_DIR, f"flux_{int(time.time())}.png")
                img.save(path); console.print(f"[green]🎨 Saved: {path}[/green]")
            except Exception as e: console.print(f"[red]Flux Error: {e}[/red]")
            return

    txt_context = ""
    for name, data in DEEP_MEMORY["files"].items():
        txt_context += f"FILE ({os.path.basename(name)}):\n{data}\n---\n"
    
    if search_data: txt_context += f"WEB RESEARCH:\n{search_data}\n---\n"

    sys_prompt = f"You are a local developer AI. Drive: {BASE_DIR}\nContext:\n{txt_context}\nTo write files, use 'SAVE_FILE: filename.ext' and end with 'END_SAVE'."
    
    # Merge system into user for DeepSeek/Chat stability (Fixes 400 Bad Request)
    msgs = chat_history + [{"role": "user", "content": f"{sys_prompt}\n\nUser Query: {user_text}"}]

    full_resp = ""
    console.print(f"\n[bold magenta]{MODEL_NAME}[/bold magenta]:")
    with Live(Markdown("💭 *Thinking...*"), console=console, auto_refresh=True) as live:
        try:
            for chunk in client.chat_completion(model=MODEL_ID, messages=msgs, stream=True, max_tokens=4000):
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    token = getattr(delta, 'reasoning_content', "") or getattr(delta, 'content', "") or ""
                    full_resp += str(token)
                    live.update(Markdown(full_resp.replace("<think>", "💭 *Thinking...*\n").replace("</think>", "\n---\n")))
        except Exception as e: 
            console.print(f"[red]Error: {e}[/red]")
            return
        
    handle_file_writing(full_resp)
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": full_resp})
    save_mem(chat_history, DEEP_MEMORY)

# --- Main Program ---
chat_history, DEEP_MEMORY = load_mem()
MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model()

while True:
    try: cmd = console.input(f"\n[bold cyan]User[/bold cyan] [dim](search/upload/status/forget/wipe/switch/exit)[/dim]: ").strip()
    except: break
    if not cmd or cmd.lower() == "exit": break
    if cmd.lower() == "switch": MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model(); continue
    if cmd.lower() == "status":
        t = Table(title="System Status"); t.add_row("Model", MODEL_NAME); t.add_row("Context Files", str(len(DEEP_MEMORY['files'])))
        console.print(t); continue
    if cmd.lower() == "wipe":
        chat_history, DEEP_MEMORY = [], {"files": {}}; os.remove(MEMORY_FILE) if os.path.exists(MEMORY_FILE) else None
        console.print("[bold red]💥 WIPED.[/bold red]"); continue
    if cmd.lower().startswith("forget"):
        file_list = list(DEEP_MEMORY["files"].keys())
        idx = console.input("[yellow]ID to remove (or 'all'): [/yellow]").strip()
        if idx.lower() == "all": DEEP_MEMORY["files"] = {}
        elif idx.isdigit() and 0 < int(idx) <= len(file_list): del DEEP_MEMORY["files"][file_list[int(idx)-1]]
        continue
    if cmd.lower().startswith("search "):
        q = cmd[7:]; res = web_search(q); run_ai(f"Research: {q}", search_data=res); continue
    if cmd.lower().startswith("upload"):
        p = os.path.abspath(os.path.expanduser(cmd[7:].strip().replace("\\ ", " ")))
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8', errors='ignore') as f: DEEP_MEMORY["files"][p] = f.read()
            console.print(f"[green]✅ Loaded {os.path.basename(p)}[/green]")
            run_ai(f"I uploaded {os.path.basename(p)}."); continue
    run_ai(cmd)



