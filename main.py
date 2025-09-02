import json
from ollama import chat, ChatResponse
from typing import List, Dict, Any
import requests, datetime
from decimal import Decimal

# Function Implementation
def get_nbs_rate(CurrencyCode: str, Date:datetime.datetime):
    """Get the rate for spacified country in particular date."""
    #datestr = datetime.datetime.strptime.(Date,"%Y-%m-%d")
    if type(Date) is str and (Date.find('current') > -1 or Date.find('now') > -1 or Date == ""):
        Date = datetime.datetime.now()
        print(type(Date))

    #if type(Date) is not datetime.datetime:
        #return None
    datestr = ""
    if type(Date) is str:
        try:
            d = datetime.datetime.strptime(Date, '%Y-%m-%d')
            datestr = Date
        except ValueError:
            return None
    
    if type(Date) is datetime.datetime or type(Date) is datetime:
        datestr = datetime.datetime.strftime(Date,"%Y-%m-%d")

    if len(datestr) < 10:
        return None

    url = f"https://nbs.sk/export/sk/exchange-rate/{datestr}/xml"
    response = requests.get(url)
    if(response.status_code != 200):
        return None
    fs = response.content.decode("utf-8").find(f'<Cube  currency="{CurrencyCode}" rate="')
    if(fs == -1):
        return None
    es = response.content.decode("utf-8").find(f'"/>',fs + 28)
    ratestr = response.content.decode("utf-8")[fs+28:es].replace(',','.')
    
    try:
        decimal_number = Decimal(ratestr)
    except ValueError:
        return None

    return json.dumps({'CurrencyCode': 'EUR', 'ExchangeRate': str(round(1/decimal_number,5))})


# Define tools for Ollama (mixed format as shown in original)
tools = [
    get_nbs_rate,  # Direct function reference
    {
        "type": "function",
        "function": {
            "name": "get_nbs_rate",
            "description": "Use this function to get the online exchange rate to the euro in one particular date. Exchange rate si available only for countries: United states of America, Japan, Czech republic, Bulgaria.",
            "parameters": {
                "CurrencyCode": {
                    "type": "string",
                    "description": "The Currency code in three digits ISO format.",
                },
                "Date": {
                    "type": "string",
                    "description": "Specific date in yyyy-mm-dd format, ",
                },                    
                "required": ["CurrencyCode","Date"],
            },
        },
    },
]

# Available functions for calling
available_functions = {
    "get_nbs_rate": get_nbs_rate
}


class OllamaReactAgent:
    """A ReAct (Reason and Act) agent using Ollama."""
    
    def __init__(self, model: str = "llama3.2"):
        self.model = model
        self.max_iterations = 10
        
    def run(self, messages: List[Dict[str, Any]]) -> str:
        """
        Run the ReAct loop until we get a final answer.
        """
        iteration = 0
        
        while iteration < self.max_iterations:
            iteration += 1
            print(f"\n--- Iteration {iteration} ---")
            
            # Call the LLM
            response: ChatResponse = chat(
                self.model,
                messages=messages,
                tools=tools,
            )
            
            print(f"LLM Response: {response.message}")
            
            # Check if there are tool calls
            if response.message.tool_calls:
                # Add the assistant's message to history
                messages.append(response.message)
                
                # Process ALL tool calls
                for tool_call in response.message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = tool_call.function.arguments
                    
                    print(f"Executing tool: {function_name}({function_args})")
                    
                    # Call the function
                    function_to_call = available_functions[function_name]
                    function_response = function_to_call(**function_args)
                    
                    print(f"Tool result: {function_response}")
                    
                    # Add tool response to messages
                    messages.append({
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(function_response),
                    })
                
                # Continue the loop to get the next response
                continue
                
            else:
                # No tool calls - we have our final answer
                final_content = response.message.content
                
                # Add the final assistant message to history
                messages.append(response.message)
                
                print(f"\nFinal answer: {final_content}")
                return final_content
        
        # If we hit max iterations, return an error
        return "Error: Maximum iterations reached without getting a final answer."


def main():
    agent = OllamaReactAgent()
    
    print("=== Example 1: Single Tool Call for country with actual exchange rate===")
    messages1 = [
        {"role": "user", "content": "What is the current exchange rate for United States of America?"}
    ]
    result1 = agent.run(messages1.copy())
    print(f"\nResult: {result1}")


    print("=== Example 2: Single Tool Call in specific date===")
    messages2 = [
        {"role": "user", "content": "What is the current exchange rates for Czech Republic on 21. of January 2021?"}
    ]
    result2 = agent.run(messages2.copy())
    print(f"\nResult: {result2}")
    
if __name__ == "__main__":
    main()