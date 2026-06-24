from ingest_homework import load_data, build_index

documents = load_data()
index = build_index(documents)

INSTRUCTIONS = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()


USER_PROMPT_TEMPLATE = """
QUESTION: {question}

CONTEXT:
{context}
""".strip()

class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=USER_PROMPT_TEMPLATE,
        model="gpt-5.4-mini"
    ):
        self.index = index
        self.llm_client = llm_client
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.model = model

    def search(question):
        boost_dict = {'question':2.0}

        return index.search(
            question,
            boost_dict=boost_dict,
            num_results=5
        )

    def build_context(search_results):
        lines = []

        for doc in search_results:
            lines.append("F: " + doc["filename"])
            lines.append("C: " + doc["content"])
            lines.append("")

        return "\n".join(lines).strip()


    def build_prompt(self, query, search_results):
        context = self.build_context(search_results)
        return self.prompt_template.format(
            question=query, context=context
        )

    def llm(self, prompt):
        input_messages = [
            {"role": "developer", "content": self.instructions},
            {"role": "user", "content": prompt}
        ]

        response = self.llm_client.responses.create(
            model=self.model,
            input=input_messages
        )

        return response.output_text
    
    def rag(self, query):
        search_results = self.search(query)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return answer    