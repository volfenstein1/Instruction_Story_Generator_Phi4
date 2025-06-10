import gradio as gr
import textstat
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

"""
Story Generator using Gradio
"""

base_model = "microsoft/Phi-4-mini-instruct"
peft_model = "volfenstein/phi4-qlora-story-generator"

model = AutoModelForCausalLM.from_pretrained(
    base_model, trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(base_model)

config = PeftConfig.from_pretrained(peft_model)
model = PeftModel.from_pretrained(model, peft_model)


def generate_story(topic, theme, wordcount, paragraphs, complexity):
    user_prompt = """Write a story which matches the following criteria:

    Topic: {topic}

    Theme: {theme}

    Wordcount: {wordcount}

    Paragraphs: {paragraphs}

    Complexity: {complexity}""".format(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        paragraphs=paragraphs,
        complexity=complexity,
    )

    # Select device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    generator = pipeline("text-generation", model=model, device=device)
    output = generator(
        [{"role": "user", "content": user_prompt}],
        max_new_tokens=512,
        return_full_text=False,
    )[0]

    return output["generated_text"]


themes = [
    "Family",
    "Deception",
    "Consciousness",
    "Growth",
    "Transformation",
    "Problem-Solving",
    "Magic",
    "Dreams",
    "Discovery",
    "Morality",
    "Coming of age",
    "Belonging",
    "Logic",
    "Celebration",
    "Planning",
    "Overcoming",
    "Friendship",
    "Honesty",
    "Helping Others",
    "Hardship",
    "The Five Senses",
    "Independence",
    "Amnesia",
    "Surprises",
    "Conscience",
    "Imagination",
    "Failure",
    "Agency",
    "Self-Acceptance",
    "Courage",
    "Hope",
    "Cooperation",
    "Humor",
    "Power",
    "Adventure",
    "Kindness",
    "Loss",
    "Strategy",
    "Curiosity",
    "Conflict",
    "Revenge",
    "Generosity",
    "Perseverance",
    "Scheming",
    "Travel",
    "Resilience",
    "Resourcefulness",
    "Teamwork",
    "Optimism",
    "Love",
]

topics = [
    "fantasy worlds",
    "hidden treasures",
    "magical objects",
    "royal kingdoms",
    "fairy tales",
    "the arts",
    "talking animals",
    "dream worlds",
    "riddles",
    "cultural traditions",
    "alien encounters",
    "subterranean worlds",
    "lost civilizations",
    "magical lands",
    "sports",
    "time travel",
    "haunted places",
    "gardens",
    "mystical creatures",
    "virtual worlds",
    "mysterious maps",
    "island adventures",
    "undercover missions",
    "unusual vehicles",
    "shape-shifting",
    "the sky",
    "school life",
    "invisibility",
    "robots and technology",
    "seasonal changes",
    "space exploration",
    "holidays",
    "sibling rivalry",
    "secret societies",
    "treasure hunts",
    "dinosaurs",
    "snowy adventures",
    "giant creatures",
    "a deadline or time limit",
    "pirates",
    "superheroes",
    "bygone eras",
    "outer space",
    "living objects",
    "lost cities",
    "enchanted forests",
    "underwater adventures",
    "miniature worlds",
]


def generate_ui(theme, topic, wordcount, paragraphs, complexity):
    story = generate_story(
        topic=topic,
        theme=theme,
        wordcount=wordcount,
        paragraphs=paragraphs,
        complexity=complexity,
    )
    wc = len(story.split())
    paras = story.count("\n") + 1 if story else 0
    grade = round(textstat.flesch_kincaid_grade(story), 1)
    metrics = f"Word count: {wc} | Paragraphs: {paras} | Flesch Kincaid Grade: {grade}"
    return story, metrics


if __name__ == "__main__":
    with gr.Blocks() as demo:
        gr.Markdown("# Story Generator")
        with gr.Row():
            theme_input = gr.Dropdown(choices=themes, label="Theme")
            topic_input = gr.Dropdown(choices=topics, label="Topic")
        wordcount_input = gr.Slider(
            50, 800, step=25, value=50, label="Target word count"
        )
        paragraphs_input = gr.Slider(
            1, 9, step=1, value=1, label="Number of paragraphs"
        )
        complexity_input = gr.Slider(
            0, 12, step=1, value=0, label="Complexity"
        )
        generate_button = gr.Button("Generate")
        story_output = gr.Textbox(lines=20, label="Generated Story")
        metrics_output = gr.Textbox(label="Metrics")
        generate_button.click(
            fn=generate_ui,
            inputs=[
                theme_input,
                topic_input,
                wordcount_input,
                paragraphs_input,
                complexity_input,
            ],
            outputs=[story_output, metrics_output],
        )
    demo.launch()
