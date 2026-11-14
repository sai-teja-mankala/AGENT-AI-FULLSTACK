import json

from django.core.management.base import BaseCommand

from agents_for_question_generation.agents.openai_agent import OpenAIQuestionAgent


class Command(BaseCommand):
    help = "Run the question-generation agent on a JSON payload file and print the result."

    def add_arguments(self, parser):
        parser.add_argument("payload", help="Path to a JSON payload file")

    def handle(self, *args, **opts):
        with open(opts["payload"], encoding="utf-8") as f:
            payload = json.load(f)
        result = OpenAIQuestionAgent().generate_questions(payload)
        self.stdout.write(json.dumps(result, indent=2, ensure_ascii=False))
