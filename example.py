# requests to finAPI
@classmethod
def get_full_financial_statement_form(cls, ticker_symbol):
    response = requests.get(f"{cls.BASE_URL}/financial-statement-full-as-reported/{ticker_symbol}?period=annual&limit=50&apikey={cls.API_KEY}")

    if response.status_code == 200:
        data = response.json()
        if isinstance(data, list) and data:
            # Without an explanation for AI like below, absolutely all matching of questions to the received data will fail
            return (f"{ticker_symbol} full financial data includes a comprehensive overview of the company's financial "
                    f"performance and position. This encompasses key financial statements such as the Income Statement, "
                    f"Balance Sheet, and Cash Flow Statement. Each statement provides a unique perspective: the Income "
                    f"Statement details revenue, expenses, and profit, highlighting operational efficiency; the Balance "
                    f"Sheet provides a snapshot of assets, liabilities, and shareholder equity, reflecting the company's "
                    f"financial stability; the Cash Flow Statement shows how cash is generated and used, indicating the "
                    f"company's liquidity. Additionally, the data may include details on debt levels, investments, "
                    f"operational costs, and other financial metrics crucial for a comprehensive financial analysis. "
                    f"Specific data for {ticker_symbol} includes: {data}")
        else:
            print(f"Data received from FinancialModelingPrep for Financial Statement: {data}")
            return None
    else:
        print(f"Error fetching data: {response.status_code}")
        return None
# all requests list...


# finAPI data loading
class MethodDataLoader:
    def __init__(self, action, ticker):
        self.action = action
        self.ticker = ticker

    def load(self):
        data = self.action(self.ticker)
        document = type('Document', (object,), {"page_content": data, "metadata": {}})()
        return [document]


# Questions data matching(yes, this will be work same with AI, it's huge and long task that was in progress)
def determine_action_and_ticker(query):
    query = query.lower()

    for ticker, company_info in COMPANIES.items():
        for name_variant in company_info["name"]:
            if name_variant in query:
                for keyword, action in company_info["actions"].items():
                    if keyword in query:
                        return action, ticker
    return None, None

COMPANIES = {
    "GOOGL": {
    "name": ["google", "googl", "alphabet"],
    "actions": {
        "balance sheet": FinancialModelingPrep.get_annual_balance_sheet,
        "stock price": FinancialModelingPrep.get_real_time_price,
        "income statement": FinancialModelingPrep.get_annual_income_statement,
        "revenue data": FinancialModelingPrep.get_annual_income_statement,
        "profit details": FinancialModelingPrep.get_annual_income_statement,
        "sales figures": FinancialModelingPrep.get_annual_income_statement,
        "cash flow": FinancialModelingPrep.get_annual_cash_flow_statement,
        "operating activities": FinancialModelingPrep.get_annual_cash_flow_statement,
        "investing activities": FinancialModelingPrep.get_annual_cash_flow_statement,
        "financing activities": FinancialModelingPrep.get_annual_cash_flow_statement,
        "profile data": FinancialModelingPrep.get_company_profile,
        "profile": FinancialModelingPrep.get_company_profile,
        "price": FinancialModelingPrep.get_real_time_price,
        "sheet": FinancialModelingPrep.get_annual_balance_sheet,
        "financials": FinancialModelingPrep.get_annual_balance_sheet,
        "info": FinancialModelingPrep.get_company_profile,
        "information": FinancialModelingPrep.get_company_profile,
        "details": FinancialModelingPrep.get_company_profile,
        "10k income": FinancialModelingPrep.get_annual_financial_statement_form,
        "10k data": FinancialModelingPrep.get_annual_financial_statement_form,
        "10q income": FinancialModelingPrep.get_quarter_financial_statement_form,
        "quarter income": FinancialModelingPrep.get_quarter_financial_statement_form,
        "10q data": FinancialModelingPrep.get_quarter_financial_statement_form,
        "full financial data": FinancialModelingPrep.get_full_financial_statement_form,
        "full data": FinancialModelingPrep.get_full_financial_statement_form,
        "cash operations": FinancialModelingPrep.get_annual_cash_flow_statement,
        "dividend info": FinancialModelingPrep.get_dividend_data,
        "payout data": FinancialModelingPrep.get_dividend_data,
        "dividend history": FinancialModelingPrep.get_dividend_data,
        "market cap": FinancialModelingPrep.get_market_capitalization,
        "company valuation": FinancialModelingPrep.get_market_capitalization,
        "earnings per share": FinancialModelingPrep.get_eps_data,
        "historical prices": FinancialModelingPrep.get_historical_stock_prices,
        "stock history": FinancialModelingPrep.get_historical_stock_prices,
        "analyst ratings": FinancialModelingPrep.get_analyst_ratings,
        "market sentiment": FinancialModelingPrep.get_analyst_ratings,
        "SEC filings": FinancialModelingPrep.get_sec_filings,
        "regulatory documents": FinancialModelingPrep.get_sec_filings,

        # etc, etc
    }
}
# all companies list...


# UI handling
@csrf_exempt
def process_query(request):
    if request.method == 'GET':
        return render(request, 'query_interface.html')

    elif request.method == 'POST':
        data = request.POST
        query = data.get('query', '')
        
        chat_history = []

        action, ticker = determine_action_and_ticker(query)

        # main logic for AI
        if action and ticker:
            loader = MethodDataLoader(action, ticker)
            data = loader.load()
            index = VectorstoreIndexCreator().from_loaders([loader])
            chain = ConversationalRetrievalChain.from_llm(
                llm=ChatOpenAI(model="gpt-3.5-turbo"),
                retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
            )

            if data:
                # Langchain used for initial question + finApi results matching
                result = chain({"question": query, "chat_history": chat_history})
                return JsonResponse({"response": result['answer']})
            else:
                return JsonResponse({"response": "Apologies, but I couldn't retrieve that information at the moment."})
        else:
            return JsonResponse({"response": "I'm not sure about the company or request type you mentioned. Can you clarify?"})

    else:
        return JsonResponse({"error": "Method not allowed"})
