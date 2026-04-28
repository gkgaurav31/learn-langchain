# PROMPT TEMPLATES

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

# Load variables from a .env file into the process environment.
# This is where your OPENAI_API_KEY is usually stored for local development.
load_dotenv()

def main():
    # This is the raw context we want the model to read.
    # In real apps, this could come from a file, database, or API response.
    information = """
    Elon Reeve Musk (born June 28, 1971) is a businessman and entrepreneur known for his leadership of Tesla, SpaceX, X, and xAI. Musk has been the wealthiest person in the world since 2025; as of April 2026, Forbes estimates his net worth to be US$809 billion.
    Born into a wealthy family in Pretoria, South Africa, Musk emigrated in 1989 to Canada; he has Canadian citizenship since his mother was born there. He received bachelor's degrees in 1997 from the University of Pennsylvania before moving to California to pursue business ventures. In 1995, Musk co-founded the software company Zip2. Following its sale in 1999, he co-founded X.com, an online payment company that later merged to form PayPal, which was acquired by eBay in 2002. Musk also became an American citizen in 2002.
    In 2002, Musk founded the space technology company SpaceX, becoming its CEO and chief engineer; the company has since led innovations in reusable rockets and commercial spaceflight. Musk joined the automaker Tesla as an early investor in 2004 and became its CEO and product architect in 2008; it has since become a leader in electric vehicles. In 2015, he co-founded OpenAI to advance artificial intelligence (AI) research, but later left; growing discontent with the organization's direction and leadership in the AI boom in the 2020s led him to establish xAI, which became a subsidiary of SpaceX in 2026. In 2022, he acquired the social network Twitter, implementing significant changes, and rebranding it as X in 2023. His other businesses include the neurotechnology company Neuralink, which he co-founded in 2016, and the tunneling company the Boring Company, which he founded in 2017. In November 2025, a Tesla pay package worth $1 trillion for Musk was approved, which he is to receive over 10 years if he meets specific goals.
    Musk is a supporter of global far-right politics, figures, and political parties. He was the largest donor in the 2024 U.S. presidential election, where he supported Donald Trump. After Trump was inaugurated as president in January 2025, Musk served as Senior Advisor to the President and as the de facto head of the Department of Government Efficiency (DOGE). Shortly before a public feud with Trump, Musk left the Trump administration in May 2025 and returned to managing his companies.
    """

    # A prompt template is a reusable "instruction pattern".
    # {information} is a placeholder that will be filled at runtime.
    summary_template = """
    Give the information {information} about a person, I want you to create:
    1. A short summary of the person
    2. Two interesting facts about the person
    """

    # ---------------------------------------------------------------
    # PromptTemplate
    # ---------------------------------------------------------------
    # A PromptTemplate is one of the core building blocks in LangChain.
    # Think of it as a "fill-in-the-blank" form for prompts:
    #
    #   - `template`        : the text sent to the model, with {placeholders}
    #                         written in curly braces (Python str.format style).
    #   - `input_variables` : the names of those placeholders. LangChain will
    #                         validate that you provide all of them at runtime
    #                         and will raise an error if any are missing or
    #                         if you pass unexpected ones.
    #
    # Why use it instead of an f-string?
    #   1. Reusability: define the prompt once, call it many times with
    #      different inputs.
    #   2. Validation: LangChain checks the variables for you.
    #   3. Composability: a PromptTemplate is a "Runnable", so it can be
    #      piped (|) into models, parsers, retrievers, etc. (see LCEL below).
    #   4. Tooling: it integrates with tracing (LangSmith), partial variables,
    #      few-shot examples, and chat-style templates.
    #
    # At runtime, calling `.invoke({"information": ...})` (directly or via a
    # chain) produces a fully formatted prompt string that is sent to the LLM.
    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template
    )

    # Create the chat model client.
    # temperature=0 makes output more deterministic (less random),
    # which is useful while learning and testing.
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    # LangChain Expression Language (LCEL):
    # "prompt | model" means "format prompt first, then send to model".
    chain = summary_prompt_template | llm

    # Invoke the chain with concrete input values for prompt variables.
    # The key name must match the placeholder name: {information}.
    response = chain.invoke(input={"information": information})

    # Chat model responses are message objects; .content is the text reply.
    print(response.content)

if __name__ == "__main__":
    main()
```
