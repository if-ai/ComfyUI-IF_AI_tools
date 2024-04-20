[![ComfyUI-IF_AI_tools](https://img.youtube.com/vi/QAnapTWnawU/0.jpg)](https://youtu.be/QAnapTWnawU?si=Uomv_NXT2n2Mg9rG)

# ComfyUI-IF_AI_tools

ComfyUI-IF_AI_tools is a set of custom nodes for ComfyUI that allows you to generate prompts using a local Large Language Model (LLM) via Ollama. 
This tool enables you to enhance your image generation workflow by leveraging the power of language models.

## Features

- [NEW] WhisperSpeech integration generate long form audio from Text while trining the voice on the fly from a 10min audio file
- https://github.com/if-ai/ComfyUI-IF_AI_WishperSpeechNode
- [NEW] ParlerTTS
- available https://github.com/if-ai/ComfyUI-IF_AI_ParlerTTSNode
- [NEW] DreamTalk generate talking avatars right inside ComfyUI
  Moved https://github.com/if-ai/ComfyUI-IF_AI_Dreamtalk/tree/main
- [NEW] Json Presets (got rid of Tetx Files)
- Use OpenAI and Claude 3 you can analize images whit the Haiku vision model
- Generate prompts using a local LLM via Ollama
- generate SD prompts or ask questions about an image with Image to prompt node
- Save generated text
- Integrate with ComfyUI for a seamless workflow

## Prerequisites
- [Ollama](https://github.com/ollama/ollama/releases) - You need to install Ollama for this tool to work. Visit [ollama.com](https://ollama.com) for more information.

Optionally Set enviromnet variables for "ANTHROPIC_API_KEY" & "OPENAI_API_KEY" with those names or otherwise it won't pick it up and the respective API keys 


## Installation
1. Install Ollama by following the instructions on their GitHub page on windows 

You can also install the Node from the ComfyUI manager 

2. Open a terminal and type following command to install the model:
   ```bash
      ollama run brxce/stable-diffusion-prompt-generator
      ```
   
4. Navigate to your ComfyUI `custom_nodes` folder, type `CMD` on the address bar to open a command prompt,
   and run the following command to clone the repository:
   ```bash
      git clone https://github.com/if-ai/ComfyUI-IF_AI_tools.git
      ```
   
5. In ComfyUI protable version just dounle click `embedded_install.bat` or  type `CMD` on the address bar on the newly created `custom_nodes\ComfyUI-IF_AI_tools` folder type 
   ```bash
      H:\ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt
      ```
   replace `C:\` for your Drive letter where you have the ComfyUI_windows_portable directory

   On custom environment activate the environment and move to the newly created ComfyUI-IF_AI_tools
   ```bash
      cd ComfyUI-IF_AI_tools
      python -m pip install -r requirements.txt
      ```
   
## Usage
1. Start ComfyUI.

2. Load the custom workflow located in the `custom_nodes\ComfyUI-IF_AI_tools\workflows` folder.

3. Run the queue to generate an image.

## Recommended Models
- [Proteus-RunDiffusion](https://huggingface.co/dataautogpt3/Proteus-RunDiffusion)
- [nous-hermes2pro](https://ollama.com/adrienbrault/nous-hermes2pro)
- [llava:7b-v1.6-mistral-q5_K_M](https://ollama.com/library/llava:7b-v1.6-mistral-q5_K_M)

## Support
If you find this tool useful, please consider supporting my work by:
- Starring the repository on GitHub: [ComfyUI-IF_AI_tools](https://github.com/if-ai/ComfyUI-IF_AI_tools)
- Subscribing to my YouTube channel: [Impact Frames](https://youtube.com/@impactframes?si=DrBu3tOAC2-YbEvc)
- Supporting me on Ko-fi: [Impact Frames Ko-fi](https://ko-fi.com/impactframes)
- Becoming a patron on Patreon: [Impact Frames Patreon](https://patreon.com/ImpactFrames)

Your support helps me bring updates and improvements faster!

## Related Tools
- [IF_prompt_MKR](https://github.com/if-ai/IF_prompt_MKR) - A similar tool available for Stable Diffusion WebUI

##
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


<img src="https://count.getloli.com/get/@IFAItools_comfy?theme=moebooru" alt=":IFAItools_comfy" />




