import sys
import argparse
from rag_langchain_nvidia_api import create_embeddings, create_chat_summarization_qa, create_chat_qa


def ask_questions(qa):
    for line in sys.stdin:
        print("Enter a question (q to quit):")
        if 'q' == line.rstrip():
            break
        query = "What is Triton?"
        result = qa({"question": query})
        print(result.get("answer"))

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--task", default="chat", help="chat, embed, chat_summarize")
args = parser.parse_args()

if args.task == 'embed':
    create_embeddings();
elif args.task == 'chat':
    qa = create_chat_qa()
    ask_questions(qa)
else:
    qa = create_chat_summarization_qa()
    ask_questions(qa)