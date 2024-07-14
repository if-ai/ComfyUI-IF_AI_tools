import textwrap

def prepare_messages(prompt, assistant_content, messages, images=None, mode=True):

    if images is not None:
        if mode:
            image_message = textwrap.dedent("""\
                Analyze the images provided, search for relevant details from the images to include on your response.
                craft a visual prompt from details extracted from the images. include keywords related framing composition, subject, action, foreground, background, visual elements, scene aesthetics and style. 
            """)
            system_message = f"{assistant_content}\n{image_message}"
            user_message = prompt if prompt.strip() != "" else "provide a visual prompt for each image provided?."
        
        else:
            image_message = textwrap.dedent("""\
                Analyze the images provided, search for relevant details from the images to include on your response.
                include relevant details extracted from the images.
            """)
            system_message = f"{assistant_content}\n{image_message}"
            user_message = prompt if prompt.strip() != "" else "describe the image in detail?."
            
        
    else:
        system_message = assistant_content
        user_message = prompt


    messages.append({"role": "user", "content": user_message})

    return user_message, system_message, messages


