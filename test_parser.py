
from typing import Optional
import pytest
import os
import unittest
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from langchain_core.exceptions import OutputParserException
from langchain_groq import ChatGroq

load_dotenv()
GROQ_API_KEY = os.getenv("GROK_API_KEY")

# 1. bring in your llm
llm = ChatGroq(
    temperature=0,
    model_name="llama3-groq-70b-8192-tool-use-preview",
    groq_api_key=GROQ_API_KEY,
)

class Stock(BaseModel):
    """Information about a company's stock"""

    symbol: str = Field(description="The stock symbol")
    name: str = Field(description="The name of the company for which the stock symbol represents")
    sector: Optional[str] = Field(default=None, description="The sector of the company")
    industry: Optional[str] = Field(default=None, description="The industry of the company")
    market_cap: Optional[int] = Field(default=None, description="The market capitalization of the company")
    # 2. implement the other fields
    # ...

    @model_validator(mode="before")
    @classmethod
    def validate_symbol_4_letters(cls, values: dict) -> dict:
        print(values)
        symbol = values['symbol']
        # 3. implement LLM validation logic
        # ...
        
        if len(symbol) != 4:
            raise ValueError("Symbol must be 4 letters long")
        return values
    
    @field_validator("market_cap", mode="before")
    @classmethod
    def validate_market_cap(cls, value: int) -> int:
        print(value)
        if value < 0:
            raise ValueError("Market cap must be greater than 0")
        return value

parser = PydanticOutputParser(pydantic_object=Stock)

prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

runnable = prompt | llm | parser


class TestParser(unittest.TestCase):
    def test_output_parser_symbol_valid(self):
        text = """
        Bank Central Asia (BBCA) is a bank in Indonesia and is part of the finance sector.
            It is in the banking industry and has a market capitalization of $8.5 billion.
        """
        # 4. implement when symbol and market cap (and other fields) are all valid
        out = runnable.invoke(text)
        assert len(out.symbol) == 4
        assert out.market_cap > 0
        assert len(out.name) > 0
        
        


    def test_output_parser_symbol_invalid(self):
        text = """
        Bank Central Asia (BCA) is a bank in Indonesia and is part of the finance sector.
            It is in the banking industry and has a market capitalization of $8.5 billion.
        """

        # assert exception is raised when the symbol is not 4 letters long
        with pytest.raises(OutputParserException):
            out = runnable.invoke(text)

    def test_output_parser_mcap_invalid(self):
        text = """
        Bank Central Asia (BBCA) is a bank in Indonesia and is part of the finance sector.
            It is in the banking industry and has a market capitalization of $-8.5 billion.
        """

        # 5. assert exception is raised when extraction task fail by detecting <0 market cap
        with pytest.raises(OutputParserException):
            out = runnable.invoke(text)
