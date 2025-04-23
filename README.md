# A (simple) script to generate Coursera-like quiz from transcripts using AI agent
The script uses Langchain to stuff the whole script into a single LLM call and generate a number of quizzes in JSON format. 

## Usage
1. Install the packages:
```
pip install -r requirements.txt
```
2. Set up API keys in secrets.toml file:

```
OPENAI_API_KEY="openai api key"
```
3. Run the code [quiz_generator.py](quiz_generator.py):
```
python quiz_generator.py --input-dir $INPUT_FOLDER --output-dir $OUTPUT_FOLDER --workers $NUMBER_OF_WORKERS(DEFAULT = 4)
```
