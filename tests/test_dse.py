from pathlib import Path

from colpali_engine.utils.torch_utils import get_torch_device

from byaldi import RAGMultiModalModel

device = get_torch_device("auto")
print(f"Using device: {device}")

path_document_1 = Path("docs/attention.pdf")
path_document_2 = Path("docs/attention_copy.pdf")

def test_model(model_type: str):
    print(f"\nTesting {model_type} model for indexing and retrieval...")

    # Initialize the model
    if model_type == "ColPali":
        model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device=device)
    elif model_type == "DSE":
        model = RAGMultiModalModel.from_pretrained("MrLight/dse-qwen2-2b-mrl-v1", model_type="DSEModel", device=device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if not Path("docs/attention.pdf").is_file():
        raise FileNotFoundError(
            f"Please download the PDF file from https://arxiv.org/pdf/1706.03762 and move it to {path_document_1}."
        )

    # Index a single PDF
    model.index(
        input_path="docs/attention.pdf",
        index_name=f"{model_type.lower()}_attention_index",
        store_collection_with_index=True,
        overwrite=True,
    )

    # Test retrieval
    queries = [
        "How does the positional encoding thing work?",
        "what's the BLEU score of this new strange method?",
    ]

    for query in queries:
        results = model.search(query, k=3)

        print(f"\nQuery: {query}")
        for result in results:
            print(
                f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}"
            )

        # Check if the expected page (6 for positional encoding) is in the top results
        if "positional encoding" in query.lower():
            assert any(
                r.page_num == 6 for r in results
            ), "Expected page 6 for positional encoding query"

        # Check if the expected pages (8 and 9 for BLEU score) are in the top results
        if "bleu score" in query.lower():
            assert any(
                r.page_num in [8, 9] for r in results
            ), "Expected pages 8 or 9 for BLEU score query"

    print(f"{model_type} model test completed.")

def test_multi_document(model_type: str):
    print(f"\nTesting {model_type} model for multi-document indexing and retrieval...")

    # Initialize the model
    if model_type == "ColPali":
        model = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2", device=device)
    elif model_type == "DSE":
        model = RAGMultiModalModel.from_pretrained("MrLight/dse-qwen2-2b-mrl-v1", model_type="DSEModel", device=device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if not Path("docs/attention.pdf").is_file() or not Path("docs/attention_copy.pdf").is_file():
        raise FileNotFoundError(
            f"Please ensure both 'attention.pdf' and 'attention_copy.pdf' are in the 'docs/' directory."
        )

    # Index a directory of documents
    model.index(
        input_path="docs/",
        index_name=f"{model_type.lower()}_multi_doc_index",
        store_collection_with_index=True,
        overwrite=True,
    )

    # Test retrieval
    queries = [
        "How does the positional encoding thing work?",
        "what's the BLEU score of this new strange method?",
    ]

    for query in queries:
        results = model.search(query, k=5)

        print(f"\nQuery: {query}")
        for result in results:
            print(
                f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}"
            )

        # Check if the expected page (6 for positional encoding) is in the top results
        if "positional encoding" in query.lower():
            assert any(
                r.page_num == 6 for r in results
            ), "Expected page 6 for positional encoding query"

        # Check if the expected pages (8 and 9 for BLEU score) are in the top results
        if "bleu score" in query.lower():
            assert any(
                r.page_num in [8, 9] for r in results
            ), "Expected pages 8 or 9 for BLEU score query"

    print(f"{model_type} model multi-document test completed.")

def test_add_to_index(model_type: str):
    print(f"\nTesting adding to an existing index for {model_type} model...")

    # Load the existing index
    if model_type == "ColPali":
        model = RAGMultiModalModel.from_index(f"{model_type.lower()}_multi_doc_index", device=device)
    elif model_type == "DSE":
        model = RAGMultiModalModel.from_index(f"{model_type.lower()}_multi_doc_index", device=device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    # Add a new document to the index
    model.add_to_index(
        input_item="docs/",
        store_collection_with_index=True,
        doc_id=[1002, 1003],
        metadata=[{"author": "John Doe", "year": 2023}] * 2,
    )

    # Test retrieval with the updated index
    queries = ["what's the BLEU score of this new strange method?"]

    for query in queries:
        results = model.search(query, k=3)

        print(f"\nQuery: {query}")
        for result in results:
            print(
                f"Doc ID: {result.doc_id}, Page: {result.page_num}, Score: {result.score}"
            )
            print(f"Metadata: {result.metadata}")

        # Check if the expected pages (8 and 9 for BLEU score) are in the top results
        if "bleu score" in query.lower():
            assert any(
                r.page_num in [8, 9] for r in results
            ), "Expected pages 8 or 9 for BLEU score query"

    print(f"{model_type} model add to index test completed.")

if __name__ == "__main__":
    print("Starting tests...")

    for model_type in ["DSE"]:
        print(f"\n\n----------------- {model_type} Model Tests -----------------")
        print(f"\n-----------------  Single PDF test  -----------------")
        test_model(model_type)

        print(f"\n-----------------  Multi document test  -----------------")
        test_multi_document(model_type)

        print(f"\n-----------------  Add to index test  -----------------")
        test_add_to_index(model_type)

    print("\nAll tests completed.")