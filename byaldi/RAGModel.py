from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from PIL import Image

from byaldi.colpali import ColPaliModel
from byaldi.dse import DSEModel
from byaldi.indexing import IndexManager
from byaldi.objects import Result


class RAGMultiModalModel:
    def __init__(
        self,
        model: Optional[Union[ColPaliModel, DSEModel]] = None,
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        self.model = model
        self.index_manager = IndexManager(index_root=index_root, verbose=verbose)
        self.device = device
        self.verbose = verbose
        self.model_type = type(model).__name__ if model else None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        model_type: str = "ColPaliModel",
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        if model_type == "ColPaliModel":
            model = ColPaliModel.from_pretrained(
                pretrained_model_name_or_path,
                device=device,
                verbose=verbose,
            )
        elif model_type == "DSEModel":
            model = DSEModel(pretrained_model_name_or_path, device=device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        instance = cls(model, index_root, device, verbose)
        instance.model_type = model_type
        return instance

    @classmethod
    def from_index(
        cls,
        index_name: str,
        index_root: str = ".byaldi",
        device: str = "cuda",
        verbose: int = 1,
    ):
        instance = cls(index_root=index_root, device=device, verbose=verbose)
        instance.index_manager.load_index(index_name)
        model_type = instance.index_manager.model_type
        model_name = instance.index_manager.model_name
        
        if model_type == "ColPaliModel":
            instance.model = ColPaliModel.from_pretrained(
                model_name,
                device=device,
                verbose=verbose,
            )
        elif model_type == "DSEModel":
            instance.model = DSEModel(model_name, device=device)
        else:
            raise ValueError(f"Unsupported model type in index: {model_type}")
        
        instance.model_type = model_type
        return instance

    def index(
        self,
        input_path: Union[str, Path],
        index_name: str,
        store_collection_with_index: bool = False,
        doc_ids: Optional[List[int]] = None,
        metadata: Optional[List[Dict[str, Union[str, int]]]] = None,
        overwrite: bool = False,
        max_image_width: Optional[int] = None,
        max_image_height: Optional[int] = None,
        **kwargs,
    ):
        self.index_manager.create_index(
            index_name,
            self.model.pretrained_model_name_or_path,
            store_collection_with_index,
            overwrite,
            max_image_width,
            max_image_height,
            model_type=self.model_type,
            model_name=self.model.pretrained_model_name_or_path,
        )
        return self.index_manager.add_to_index(
            input_path,
            self.model.encode_image,
            doc_ids,
            metadata,
        )

    def add_to_index(
        self,
        input_item: Union[str, Path, Image.Image],
        store_collection_with_index: bool = False,
        doc_id: Optional[int] = None,
        metadata: Optional[Dict[str, Union[str, int]]] = None,
    ):
        return self.index_manager.add_to_index(
            input_item,
            self.model.encode_image,
            doc_id,
            metadata,
        )

    def search(
        self,
        query: Union[str, List[str]],
        k: int = 10,
        return_base64_results: Optional[bool] = None,
    ) -> Union[List[Result], List[List[Result]]]:
        return self.index_manager.search(
            query,
            self.model.score,
            k,
            return_base64_results,
        )

    def get_doc_ids_to_file_names(self):
        return self.index_manager.get_doc_ids_to_file_names()

    def as_langchain_retriever(self, **kwargs: Any):
        from byaldi.integrations import ByaldiLangChainRetriever
        return ByaldiLangChainRetriever(model=self, kwargs=kwargs)