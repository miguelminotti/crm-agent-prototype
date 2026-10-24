"""
Dataset for evaluating the CRM agent interactions.

`tools_called` and `must_not_call` are used to specify which tools should or should not be called during
the evaluation. Both means "equals", i.e., the tool must be called exactly as specified.
"""

from typing import Any

CRM_AGENT_EVAL_DATASET: list[dict[str, Any]] = [
    {
        # Scenario 1
        # Create a company with formatted CNPJ
        "metadata": {"scenario": "formatted_cnpj"},
        "inputs": {"message": "Crie a empresa Acme com CNPJ 12.345.678/0001-90"},
        "reference_outputs": {
            "companies_created": [{"name": "Acme", "cnpj": "12345678000190"}],
            "tools_called": ["find_companies", "create_company"],
        },
    },
    {
        # Scenario 2
        # Create a contact linked to an existent company (with seed)
        "metadata": {"scenario": "link_to_existent_company"},
        "inputs": {
            "message": "Crie o contato João da Silva com email joao@acme.com e vincule à empresa Acme",
            "seed": {"companies": [{"id": "c-1", "name": "Acme", "cnpj": "12345678000190"}]},
        },
        "reference_outputs": {
            "contacts_created": [{"name": "João da Silva", "email": "joao@acme.com", "company_id": "c-1"}],
            "tools_called": ["find_companies", "create_contact"],
        },
    },
    {
        # Scenario 3
        # Create a contact linked to a non-existent company
        "metadata": {"scenario": "link_to_nonexistent_company"},
        "inputs": {"message": "Crie o contato Maria Oliveira com email maria@globex.com e vincule à empresa Globex"},
        "reference_outputs": {
            "contacts_created": [],
            "tools_called": ["find_companies"],
            "must_not_call": ["create_contact", "create_company"],
        },
    },
    {
        # Scenario 4
        # Ambiguous: seed with "Acme LTDA" and "Acme Brasil", try linking to "Acme"
        "metadata": {"scenario": "ambiguous_company_link"},
        "inputs": {
            "message": "Crie o contato Pedro Souza com email pedro@acme.com e vincule à empresa Acme",
            "seed": {
                "companies": [
                    {"id": "c-1", "name": "Acme LTDA", "cnpj": "12345678000191"},
                    {"id": "c-2", "name": "Acme Brasil", "cnpj": "12345678000192"},
                ]
            },
        },
        "reference_outputs": {
            "contacts_created": [],
            "tools_called": ["find_companies"],
            "must_not_call": ["create_contact"],
        },
    },
    {
        # Scenario 5
        # Try creating a company with duplicated CNPJ
        "metadata": {"scenario": "duplicated_cnpj"},
        "inputs": {
            "message": "Crie a empresa Acme com CNPJ 12.345.678/0001-90",
            "seed": {"companies": [{"id": "c-1", "name": "Acme", "cnpj": "12345678000190"}]},
        },
        "reference_outputs": {
            "companies_created": [],
            "tools_called": ["find_companies"],
            "must_not_call": ["create_company", "create_contact"],
        },
    },
    {
        # Scenario 6
        # Readonly user query ("Quais empresas temos com a letra A?") does not write any data
        "metadata": {"scenario": "readonly_query"},
        "inputs": {
            "message": "Quais empresas temos com a letra A?",
            "seed": {
                "companies": [
                    {"id": "c-1", "name": "Acme", "cnpj": "12345678000190"},
                    {"id": "c-2", "name": "Alpha", "cnpj": "12345678000191"},
                    {"id": "c-3", "name": "Beta", "cnpj": "12345678000192"},
                ]
            },
        },
        "reference_outputs": {
            "companies_created": [],
            "tools_called": ["find_companies"],
            "must_not_call": ["create_company", "create_contact"],
        },
    },
]
