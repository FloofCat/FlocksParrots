import openai
import backoff 

class Inference:
    def __init__(self, config):
        self.config = config
        self.api_key = self.config["OPENAI_KEY"]
        
        # Create a client
        self.client = openai.Client(api_key=self.api_key)
    
    def query(self, instruction, example, prompt):
        chat = [
            {"role": "system", "content": "You are a AI assistant that is designed to answer user's queries."},
            {"role": "user", "content": f"{instruction}\n\n{example}\n\n{prompt}"}
        ]
        
        @backoff.on_exception(backoff.expo, BaseException)
        def completions_with_backoff(prompt):
            response = self.client.chat.completions.create(
                model="babbage-002",
                messages=prompt,
                max_tokens=1024,
                temperature=0.7,
                logprobs=True,
                top_logprobs=5
            )
            return response.choices[0].message.content, response.choices[0].logprobs.content.logprob

        return completions_with_backoff(chat)