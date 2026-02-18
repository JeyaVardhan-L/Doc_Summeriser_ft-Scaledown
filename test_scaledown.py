import os
from dotenv import load_dotenv
from scaledown.compressor.scaledown_compressor import ScaleDownCompressor

# load env vars from .env file
load_dotenv()

api_key=os.getenv("SCALEDOWN_API_KEY")

if api_key is None:
    raise ValueError("API key not found")
# create compressor object
# this talks to scaledown servers
compressor=ScaleDownCompressor(
    target_model = "gpt-4o",
    api_key=api_key
)
context="""
Lionel Andrés Messi (Spanish pronunciation: [ljoˈnel anˈdɾes ˈmesi]; born 24 June 1987), also known as Leo Messi, is an Argentine professional footballer who plays as a forward for Ligue 1 club Paris Saint-Germain and captains the Argentina national team. Widely regarded as one of the greatest players of all time, Messi has won a record seven Ballon d'Or awards,[note 2] a record six European Golden Shoes, and in 2020 was named to the Ballon d'Or Dream Team.
...
Later that year, Messi became the second footballer and second team-sport athlete to surpass $1 billion in career earnings.
"""

prompt="How many Ballon d'Or awards did Messi win?"

# send text + question to scaledown
result=compressor.compress(
    context = context,
    prompt=prompt
)

print("\nCompressed Content:")
print(result.content)
print("\nMetrics:")
print("Original Tokens:",result.tokens[0])
print("Compressed Tokens:", result.tokens[1])
print("Compression Ratio:",result.compression_ratio)
print("Savings Percent:", result.savings_percent)
print("Latency (ms):",result.latency)
print("Model Used:", result.model)
