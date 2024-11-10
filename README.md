[![ComfyUI-IF_AI_tools](https://img.youtube.com/vi/QAnapTWnawU/0.jpg)](https://youtu.be/QAnapTWnawU?si=Uomv_NXT2n2Mg9rG)

# ComfyUI-IF_AI_tools

ComfyUI-IF_AI_tools is a set of custom nodes to Run Local and API LLMs and LMMs, Features OCR-RAG (Bialdy), nanoGraphRAG, Supervision Object Detection, supports Ollama, LlamaCPP LMstudio, Koboldcpp, TextGen, Transformers or via APIs Anthropic, Groq, OpenAI, Google Gemini, Mistral, xAI and create your own charcters assistants (SystemPrompts) with custom presets and muchmore


# Prerequisite Installation (Poppler)

To ensure compatibility and functionality with all tools, you may need `poppler` for PDF-related operations. Use `scoop` to install `poppler` on Windows:

### Step 1: Install `scoop` (if not already installed)
If you haven't installed `scoop` yet, run the following command in **PowerShell**:

```powershell
iwr -useb get.scoop.sh | iex
```
Step 2: Install poppler with scoop
Once scoop is installed, you can install poppler by running:

windows 10+ istall scoop and then 
```powershell
scoop install poppler
```
Debian/Ubuntu
```bash
sudo apt-get install poppler-utils
```
MacOS
```bash
brew install poppler
```

check is working
```powershell
pdftotext -v
```

### Install Ollama

You can technically use any LLM API that you want, but for the best expirience install Ollama and set it up.
- Visit [ollama.com](https://ollama.com) for more information.

To install Ollama models just open CMD or any terminal and type the run command follow by the model name such as
```powershell
ollama run llama3.2-vision
```
If you want to use omost 
```bash
ollama run impactframes/dolphin_llama3_omost
```
if you need a good smol model
```bash
ollama run ollama run llama3.2
```

Optionally Set enviromnet variables for any of your favourite LLM API keys "XAI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY", "OPENAI_API_KEY" or "GROQ_API_KEY" with those names or otherwise
it won't pick it up you can also use .env file to store your keys

## Features
_[NEW]_ nanoGraphRAG, 
_[NEW]_ OCR-RAG ColPali & ColQwen (Bialdy), 
_[NEW]_ Supervision Object Detection Florence2   
_[NEW]_ Endpoints xAI, Transformers,
_[NEW]_ IF_Assistants System Prompts with Reasoning/Reflection/Reward Templates and custom presets

- Gemini, Groq, Mistral, OpenAI, Anthropic, Google, xAI, Transformers, Koboldcpp, TextGen, LlamaCPP, LMstudio, Ollama 
- Omost_tool the first tool 
- Vision Models Haiku, Florence2
- [Ollama-Omost]https://ollama.com/impactframes/dolphin_llama3_omost can be 2x to 3x faster than other Omost Models
LLama3 and Phi3 IF_AI Prompt mkr models released
![ComfyUI_00021_](https://github.com/if-ai/ComfyUI-IF_AI_tools/assets/21185218/fac9fb38-66ac-431b-8ef9-b0fee5d0e5dc)

`ollama run impactframes/llama3_ifai_sd_prompt_mkr_q4km:latest`

`ollama run impactframes/ifai_promptmkr_dolphin_phi3:latest`

https://huggingface.co/impactframes/llama3_if_ai_sdpromptmkr_q4km

https://huggingface.co/impactframes/ifai_promptmkr_dolphin_phi3_gguf


## Installation
1. Open the manager search for IF_AI_tools and install

### Install ComfyUI-IF_AI_tools -hardest way
   
1. Navigate to your ComfyUI `custom_nodes` folder, type `CMD` on the address bar to open a command prompt,
   and run the following command to clone the repository:
   ```bash
      git clone https://github.com/if-ai/ComfyUI-IF_AI_tools.git
      ```
OR
1. In ComfyUI protable version just dounle click `embedded_install.bat` or  type `CMD` on the address bar on the newly created `custom_nodes\ComfyUI-IF_AI_tools` folder type 
   ```bash
      H:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
      ```
   replace `C:\` for your Drive letter where you have the ComfyUI_windows_portable directory

2. On custom environment activate the environment and move to the newly created ComfyUI-IF_AI_tools
   ```bash
      cd ComfyUI-IF_AI_tools
      python -m pip install -r requirements.txt
      ```

## Related Tools
- [IF_prompt_MKR](https://github.com/if-ai/IF_prompt_MKR) 
-  A similar tool available for Stable Diffusion WebUI

## Videos
AIFuzz made a great video usining ollama and IF_AI tools 

[![AIFuzz](https://img.youtube.com/vi/nZx5g3TGsNc/0.jpg)](https://youtu.be/nZx5g3TGsNc?si=DFIqFuPoyKY1qJ2n)

Also Future thinker @ Benji Thankyou both for putting out this awesome videos

[![Future Thinker @Benji](https://img.youtube.com/vi/EQZWyn9eCFE/0.jpg)](https://youtu.be/EQZWyn9eCFE?si=jgC28GL7bwFWj_sK)


## Example using normal Model
ancient Megastructure, small lone figure 
'A dwarfed figure standing atop an ancient megastructure, worn stone towering overhead. Underneath the dim moonlight, intricate engravings adorn the crumbling walls. Overwhelmed by the sheer size and age of the structure, the small figure appears lost amidst the weathered stone behemoth. The background reveals a dark landscape, dotted with faint twinkles from other ancient structures, scattered across the horizon. The silent air is only filled with the soft echoes of distant whispers, carrying secrets of times long past. ethereal-fantasy-concept-art, magical-ambiance, magnificent, celestial, ethereal-lighting, painterly, epic, majestic, dreamy-atmosphere, otherworldly, mystic-elements, surreal, immersive-detail'
![_IF_prompt_Mkr__00011_](https://github.com/if-ai/ComfyUI-IF_AI_tools/assets/21185218/08dde522-f541-49f4-aa6b-e0653f13aa52)
![_IF_prompt_Mkr__00012_](https://github.com/if-ai/ComfyUI-IF_AI_tools/assets/21185218/ec3ef715-fbe6-4ba0-80f8-00bf10f56f7b)
![_IF_prompt_Mkr__00010_](https://github.com/if-ai/ComfyUI-IF_AI_tools/assets/21185218/e4dc671b-8eea-47f3-84ef-876e5938e120)
![_IF_prompt_Mkr__00014_](https://github.com/if-ai/ComfyUI-IF_AI_tools/assets/21185218/d0b436cd-c4a8-41a2-83ad-34d8c50bb39b)

## TODO
- [ ] Fix Bugs and make it work on latest ComfyUI
- [ ] Fix Graph Visualizer Node
- [ ] Tweak IF_Assistants and Templates
- [ ] FrontEnd for IF_Assistants and Chat 
- [ ] Node and workflow creator
- [ ] two additional Endpoints one API and one Local
- [ ] Add New workflows
- [ ] Image Generation, Text 2 Image, Image to Image, Video Generation

## Support
If you find this tool useful, please consider supporting my work by:
- Starring the repository on GitHub: [ComfyUI-IF_AI_tools](https://github.com/if-ai/ComfyUI-IF_AI_tools)
- Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/@impactframes?si=DrBu3tOAC2-YbEvc)
- Follow me on X: [Impact Frames X](https://x.com/impactframesX)
- Supporting me on Ko-fi: [Impact Frames Ko-fi](https://ko-fi.com/impactframes)
- Becoming a patron on Patreon: [Impact Frames Patreon](https://patreon.com/ImpactFrames)
Thank You!

<img src="https://count.getloli.com/get/@IFAItools_comfy?theme=moebooru" alt=":IFAItools_comfy" />




