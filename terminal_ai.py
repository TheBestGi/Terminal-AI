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
    "3": ("Qwen2-VL (Vision)", "Qwen/Qwen2-VL-7B-Instruct", "vision"),
    "4": ("Flux.1 Schnell (Art)", "black-forest-labs/FLUX.1-schnell", "image"),
    "5": ("Qwen 2.5 Coder 32B", "Qwen/Qwen2.5-Coder-32B-Instruct", "chat")
}

# --- Core Logic ---
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
            json.dump({"history": history[-15:], "deep_mem": deep_mem}, f)
    except: pass

def fix_path(p):
    return os.path.abspath(os.path.expanduser(p.strip().strip("'\"").replace("\\ ", " ")))

def select_model():
    console.clear()
    console.print(Panel("[bold cyan]🛠️ TERMINAL AI: ULTIMATE EDITION[/bold cyan]"))
    for k, (n, _, t) in MODELS.items():
        console.print(f"[bold magenta][{k}][/bold magenta] {n} [dim]({t})[/dim]")
    return MODELS.get(console.input("\nBrain ID: ").strip(), MODELS["1"])

def handle_file_writing(text):
    pattern = r"SAVE_FILE:\s*([a-zA-Z0-9_\-\.]+)\s*\n(.*?)(?=SAVE_FILE:|END_SAVE|$)"
    matches = re.findall(pattern, text, re.DOTALL)
    for filename, content in matches:
        fpath = os.path.join(BASE_DIR, filename.strip())
        clean = re.sub(r"```[a-z]*\n", "", content).replace("```", "").strip()
        try:
            with open(fpath, "w") as f: f.write(clean)
            console.print(Panel(f"[bold green]💾 AUTO-SAVED:[/bold green] {fpath}", border_style="green"))
        except Exception as e: console.print(f"[red]Write Error: {e}[/red]")

# --- Execution ---
chat_history, DEEP_MEMORY = load_mem()
MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model()

def run_ai(user_text, search_data=None):
    global chat_history
    if MODEL_TYPE == "image":
        with console.status("[bold magenta]Fluxing..."):
            img = client.text_to_image(user_text, model=MODEL_ID)
            path = os.path.join(IMAGE_DIR, f"flux_{int(time.time())}.png")
            img.save(path); console.print(f"[green]🎨 Saved: {path}[/green]"); return

    txt_context, img_list = "", []
    for name, data in DEEP_MEMORY["files"].items():
        if str(data).startswith("data:image"):
            img_list.append({"type": "image_url", "image_url": {"url": data}})
        else:
            txt_context += f"FILE ({os.path.basename(name)}):\n{data}\n---\n"
    
    if search_data: txt_context += f"WEB RESEARCH:\n{search_data}\n---\n"

    sys_prompt = f"You are a local dev AI. Drive: {BASE_DIR}\nContext:\n{txt_context}\nTo write files, use 'SAVE_FILE: filename.ext' and end with 'END_SAVE'."
    user_content = [{"type": "text", "text": user_text}] + img_list if (MODEL_TYPE == "vision" and img_list) else user_text
    msgs = [{"role": "system", "content": sys_prompt}] + chat_history + [{"role": "user", "content": user_content}]
    
    full_resp = ""
    console.print(f"\n[bold magenta]{MODEL_NAME}[/bold magenta]:")
    live = Live(Markdown("💭 *Thinking...*"), console=console, auto_refresh=True)
    try:
        live.start()
        for chunk in client.chat_completion(model=MODEL_ID, messages=msgs, stream=True, max_tokens=4000):
            if chunk.choices:
                delta = chunk.choices[0].delta
                token = getattr(delta, 'reasoning_content', "") or getattr(delta, 'content', "") or ""
                full_resp += str(token)
                live.update(Markdown(full_resp.replace("<think>", "💭 *Thinking...*\n").replace("</think>", "\n---\n")))
    finally:
        live.stop()
        
    handle_file_writing(full_resp)
    chat_history.append({"role": "user", "content": user_text})
    chat_history.append({"role": "assistant", "content": full_resp})
    save_mem(chat_history, DEEP_MEMORY)

while True:
    try: cmd = console.input(f"\n[bold cyan]User[/bold cyan] [dim](search/upload/status/wipe/switch/exit)[/dim]: ").strip()
    except: break
    if not cmd or cmd.lower() == "exit": break
    if cmd.lower() == "switch": MODEL_NAME, MODEL_ID, MODEL_TYPE = select_model(); continue
    if cmd.lower() == "status":
        t = Table(title="System Status"); t.add_row("Model", MODEL_NAME); t.add_row("Files", str(len(DEEP_MEMORY['files'])))
        console.print(t); continue
    if cmd.lower() == "wipe":
        chat_history, DEEP_MEMORY = [], {"files": {}}
        if os.path.exists(MEMORY_FILE): os.remove(MEMORY_FILE)
        console.print("[bold red]💥 MEMORY WIPED.[/bold red]"); continue
    
    if cmd.lower().startswith("search "):
        with DDGS() as ddgs:
            res = "\n".join([f"{r['href']}: {r['body']}" for r in ddgs.text(cmd[7:], max_results=5)])
        run_ai(f"Research: {cmd[7:]}", search_data=res); continue

    if cmd.lower().startswith("upload"):
        p = fix_path(cmd[7:].strip() or console.input("[yellow]Path: [/yellow]"))
        if os.path.exists(p):
            if p.lower().endswith(('.png', '.jpg', '.jpeg')):
                with open(p, "rb") as i: DEEP_MEMORY["files"][p] = f"data:image/jpeg;base64,{base64.b64encode(i.read()).decode('utf-8')}"
            else:
                with open(p, 'r', encoding='utf-8', errors='ignore') as f: DEEP_MEMORY["files"][p] = f.read()
            console.print(f"[green]✅ Loaded {os.path.basename(p)}[/green]")
            run_ai(f"Acknowledge {os.path.basename(p)}."); continue
    run_ai(cmd)
