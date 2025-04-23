import os
import json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import toml

config = toml.load("secrets.toml")
os.environ["OPENAI_API_KEY"] = config["OPENAI_API_KEY"]

def load_transcripts(folder_path: str):
    transcripts = []
    for fn in os.listdir(folder_path):
        if fn.lower().endswith(".txt"):
            path = os.path.join(folder_path, fn)
            with open(path, "r", encoding="utf-8") as f:
                transcripts.append((fn, f.read()))
    return transcripts

def process_transcript(filename: str, text: str, quiz_chain, output_dir: str) -> tuple:
    """
    Process a single transcript: generate quiz, write JSON file, return status.
    Returns (filename, out_path, error_message).
    """
    base = os.path.splitext(filename)[0]
    transcript_id = base.upper()
    doc = Document(page_content=text)
    try:
        raw = quiz_chain.invoke({
            "transcript_id": transcript_id,
            "context": [doc]
        })
        
        raw = raw.replace("```json", "").replace("```", "").strip()
        obj = json.loads(raw)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{transcript_id}.json")
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump(obj, out_f, ensure_ascii=False, indent=2)
        return (filename, out_path, None)

    except json.JSONDecodeError as e:
        return (filename, None, f"JSON parse error: {e}")
    except Exception as e:
        return (filename, None, str(e))


def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple-choice quizzes from transcript text files."
    )
    parser.add_argument(
        '-i', '--input-dir',
        default='texts',
        help='Directory containing transcript .txt files'
    )
    parser.add_argument(
        '-o', '--output-dir',
        default='quiz',
        help='Directory to write output JSON quizzes'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Number of worker threads for parallel processing'
    )
    args = parser.parse_args()

    transcripts_dir = args.input_dir
    output_dir = args.output_dir
    max_workers = args.workers

    transcripts = load_transcripts(transcripts_dir)
    if not transcripts:
        print(f"No transcript files found in '{transcripts_dir}'.")
        return

    llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2)
    prompt = PromptTemplate(
        input_variables=["transcript_id","context"],
        template="""
You are an AI expert that generates quiz questions based on a transcript.
Your task is to create multiple-choice questions that test the reader's understanding of the content (In Coursera Style).
The questions should be clear, concise, and relevant to the material in the transcript.
Generate a **single** JSON object with this exact schema:

{{
  "transcript_id": "<must equal the given transcript_id>",
  "title": "<a concise (5–8 word) summary of the transcript>",
  "quizzes": [
    {{
      "question_id": "Q1",
      "question": "<question stem>",
      "options": ["opt A", "opt B", "opt C", "opt D"],
      "correct_answer_index": <0–3>,
      "explanation": "<provide a brief explanation of the correct answer, must be inferrable from the transcript>"
    }}
    … at least 3 such entries …
  ]
}}

Requirements:
- Questions to support factual recall among diverse set of general learners.
- `transcript_id` field **must** equal "{transcript_id}".
- At least **3** questions.
- Exactly **4** options per question (1 correct, 3 plausible distractors that are relevant, challenging, and non-random).
- Zero-based indexing for `correct_answer_index`.
- **Respond with JSON only**—no extra text.

Transcript text:
{context}

""",
    )

    quiz_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context",
    )

    os.makedirs(output_dir, exist_ok=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_transcript, fn, text, quiz_chain, output_dir): fn
            for fn, text in transcripts
        }
        for future in as_completed(futures):
            fn = futures[future]
            filename, out_path, err = future.result()
            if err:
                print(f"Error processing {fn}: {err}")
            else:
                print(f"Generated quiz for {fn} → {out_path}")

if __name__ == "__main__":
    main()



