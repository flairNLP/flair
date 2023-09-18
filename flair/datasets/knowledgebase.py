import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Union

import flair
from flair.data import Concept
from flair.file_utils import cached_path, unpack_file


class KnowledgebaseLinkingDictionary:
    """Base class for downloading and reading of dictionaries for knowledgebase entity linking.

    A dictionary represents all entities of a knowledge base and their associated ids.
    """

    def __init__(
        self,
        candidates: Iterable[Concept],
        dataset_name: Optional[str] = None,
    ):
        """Initialize the Knowledgebase linking dictionary.

        Args:
            candidates: A iterable sequence of all Candidates contained in the knowledge base.
        """
        # this dataset name
        if dataset_name is None:
            dataset_name = self.__class__.__name__.lower()
        self._dataset_name = dataset_name

        candidates = list(candidates)

        self._idx_to_candidates = {candidate.concept_id: candidate for candidate in candidates}
        self._text_to_index = {
            text: candidate.concept_id
            for candidate in candidates
            for text in [candidate.concept_name, *candidate.synonyms]
        }

    @property
    def database_name(self) -> str:
        """Name of the database represented by the dictionary."""
        return self._dataset_name

    @property
    def text_to_index(self) -> Dict[str, str]:
        return self._text_to_index

    @property
    def candidates(self) -> List[Concept]:
        return list(self._idx_to_candidates.values())

    def __getitem__(self, item: str) -> Concept:
        return self._idx_to_candidates[item]


class HunerEntityLinkingDictionary(KnowledgebaseLinkingDictionary):
    """Base dictionary with data already in huner format.

    Every line in the file must be formatted as follows:

        concept_id||concept_name

    If multiple concept ids are associated to a given name they have to be separated by a `|`, e.g.

        7157||TP53|tumor protein p53
    """

    def __init__(self, path: Union[str, Path], dataset_name: str):
        self.dataset_file = Path(path)
        self._dataset_name = dataset_name
        super().__init__(self._load_candidates(), dataset_name=dataset_name)

    def _load_candidates(self):
        with open(self.dataset_file) as fp:
            for line in fp:
                line = line.strip()
                if line == "":
                    continue
                assert "||" in line, "Preprocessed EntityLinkingDictionary must have lines in the format: `cui||name`"
                cui, name = line.split("||", 1)
                name = name.lower()
                cui, *additional_ids = cui.split("|")
                yield Concept(
                    concept_id=cui,
                    concept_name=name,
                    database_name=self._dataset_name,
                    additional_ids=additional_ids,
                )


class CTD_DISEASES_DICTIONARY(KnowledgebaseLinkingDictionary):
    """Dictionary for named entity linking on diseases using the Comparative Toxicogenomics Database (CTD).

    Fur further information can be found at https://ctdbase.org/
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
    ):
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        base_path = Path(base_path)

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        data_file = self.download_dictionary(data_folder)

        super().__init__(self.parse_file(data_file), dataset_name="CTD-DISEASES")

    def download_dictionary(self, data_dir: Path) -> Path:
        result_file = data_dir / "CTD_diseases.tsv"
        data_url = "https://ctdbase.org/reports/CTD_diseases.tsv.gz"

        if not result_file.exists():
            data_path = cached_path(data_url, data_dir)
            unpack_file(data_path, unpack_to=result_file, keep=False)

        return result_file

    def parse_file(self, original_file: Path) -> Iterator[Concept]:
        columns = [
            "symbol",
            "identifier",
            "alternative_identifiers",
            "definition",
            "parent_identifiers",
            "tree_numbers",
            "parent_tree_numbers",
            "synonyms",
            "slim_mappings",
        ]

        with open(original_file, encoding="utf-8") as f:
            reader = csv.DictReader(filter(lambda r: r[0] != "#", f), fieldnames=columns, delimiter="\t")

            for row in reader:
                identifier = row["identifier"]
                additional_identifiers = [i for i in row.get("alternative_identifiers", "").split("|") if i != ""]

                if identifier == "MESH:C" and not additional_identifiers:
                    return None

                symbol = row["symbol"]

                synonyms = [s for s in row.get("synonyms", "").split("|") if s != ""]
                definition = row["definition"]

                yield Concept(
                    concept_id=identifier,
                    concept_name=symbol,
                    database_name="CTD-DISEASES",
                    additional_ids=additional_identifiers,
                    synonyms=synonyms,
                    description=definition,
                )


class CTD_CHEMICALS_DICTIONARY(KnowledgebaseLinkingDictionary):
    """Dictionary for named entity linking on chemicals using the Comparative Toxicogenomics Database (CTD).

    Fur further information can be found at https://ctdbase.org/
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
    ):
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        base_path = Path(base_path)

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        data_file = self.download_dictionary(data_folder)

        super().__init__(self.parse_file(data_file), dataset_name="CTD-CHEMICALS")

    def download_dictionary(self, data_dir: Path) -> Path:
        result_file = data_dir / "CTD_chemicals.tsv"
        data_url = "https://ctdbase.org/reports/CTD_chemicals.tsv.gz"

        if not result_file.exists():
            data_path = cached_path(data_url, data_dir)
            unpack_file(data_path, unpack_to=result_file)

        return result_file

    def parse_file(self, original_file: Path) -> Iterator[Concept]:
        columns = [
            "symbol",
            "identifier",
            "casrn",
            "definition",
            "parent_identifiers",
            "tree_numbers",
            "parent_tree_numbers",
            "synonyms",
        ]

        with open(original_file, encoding="utf-8") as f:
            reader = csv.DictReader(filter(lambda r: r[0] != "#", f), fieldnames=columns, delimiter="\t")

            for row in reader:
                identifier = row["identifier"]
                additional_identifiers = [i for i in row.get("alternative_identifiers", "").split("|") if i != ""]

                if identifier == "MESH:D013749":
                    # This MeSH ID was used by MeSH when this chemical was part of the MeSH controlled vocabulary.
                    continue

                symbol = row["symbol"]

                synonyms = [s for s in row.get("synonyms", "").split("|") if s != "" and s != symbol]
                definition = row["definition"]

                yield Concept(
                    concept_id=identifier,
                    concept_name=symbol,
                    database_name="CTD-CHEMICALS",
                    additional_ids=additional_identifiers,
                    synonyms=synonyms,
                    description=definition,
                )


class NCBI_GENE_HUMAN_DICTIONARY(KnowledgebaseLinkingDictionary):
    """Dictionary for named entity linking on diseases using the NCBI Gene ontology.

    Note that this dictionary only represents human genes - gene from different species
    aren't included!

    Fur further information can be found at https://www.ncbi.nlm.nih.gov/gene/
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
    ):
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        base_path = Path(base_path)

        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        data_file = self.download_dictionary(data_folder)

        super().__init__(self.parse_dictionary(data_file), dataset_name="NCBI-GENE-HUMAN")

    def _is_invalid_name(self, name: Optional[str]) -> bool:
        """Determine if a name should be skipped."""
        if name is None:
            return False
        name = name.strip()
        EMPTY_ENTRY_TEXT = [
            "when different from all specified ones in Gene.",
            "Record to support submission of GeneRIFs for a gene not in Gene",
        ]

        newentry = name == "NEWENTRY"
        empty = name == ""
        minus = name == "-"
        text_comment = any(e in name for e in EMPTY_ENTRY_TEXT)

        return any([newentry, empty, minus, text_comment])

    def download_dictionary(self, data_dir: Path) -> Path:
        result_file = data_dir / "Homo_sapiens.gene_info"
        data_url = "https://ftp.ncbi.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz"

        if not result_file.exists():
            data_path = cached_path(data_url, data_dir)
            unpack_file(data_path, unpack_to=result_file)

        return result_file

    def parse_dictionary(self, original_file: Path) -> Iterator[Concept]:
        synonym_fields = (
            "Symbol_from_nomenclature_authority",
            "Full_name_from_nomenclature_authority",
            "description",
            "Synonyms",
            "Other_designations",
        )
        field_names = [
            "tax_id",
            "GeneID",
            "Symbol",
            "LocusTag",
            "Synonyms",
            "dbXrefs",
            "chromosome",
            "map_location",
            "description",
            "type_of_gene",
            "Symbol_from_nomenclature_authority",
            "Full_name_from_nomenclature_authority",
            "Nomenclature_status",
            "Other_designations",
            "Modification_date",
            "Feature_type",
        ]

        with open(original_file, encoding="utf-8") as f:
            reader = csv.DictReader(filter(lambda r: r[0] != "#", f), fieldnames=field_names, delimiter="\t")

            for row in reader:
                identifier = row["GeneID"]
                symbol = row["Symbol"]

                if self._is_invalid_name(symbol):
                    continue
                additional_identifiers = [i for i in row.get("alternative_identifiers", "").split("|") if i != ""]

                if identifier == "MESH:D013749":
                    # This MeSH ID was used by MeSH when this chemical was part of the MeSH controlled vocabulary.
                    continue

                synonyms = []
                for synonym_field in synonym_fields:
                    synonyms.extend([name.replace("'", "") for name in row.get(synonym_field, "").split("|")])
                synonyms = sorted([sym for sym in set(synonyms) if not self._is_invalid_name(sym)])
                if symbol in synonyms:
                    synonyms.remove(symbol)

                yield Concept(
                    concept_id=identifier,
                    concept_name=symbol,
                    database_name="NCBI-GENE-HUMAN",
                    additional_ids=additional_identifiers,
                    synonyms=synonyms,
                )


class NCBI_TAXONOMY_DICTIONARY(KnowledgebaseLinkingDictionary):
    """Dictionary for named entity linking on organisms / species using the NCBI taxonomy ontology.

    Further information about the ontology can be found at https://www.ncbi.nlm.nih.gov/taxonomy
    """

    def __init__(
        self,
        base_path: Optional[Union[str, Path]] = None,
    ):
        if base_path is None:
            base_path = flair.cache_root / "datasets"
        base_path = Path(base_path)
        dataset_name = self.__class__.__name__.lower()

        data_folder = base_path / dataset_name

        data_file = self.download_dictionary(data_folder)

        super().__init__(self.parse_dictionary(data_file), dataset_name="NCBI-TAXONOMY")

    def download_dictionary(self, data_dir: Path) -> Path:
        result_file = data_dir / "names.dmp"
        data_url = "https://ftp.ncbi.nih.gov/pub/taxonomy/new_taxdump/new_taxdump.tar.gz"

        if not result_file.exists():
            data_path = cached_path(data_url, data_dir)
            unpack_file(data_path, unpack_to=result_file)

        return result_file

    def parse_dictionary(self, original_file: Path) -> Iterator[Concept]:
        ncbi_taxonomy_synset = [
            "genbank common name",
            "common name",
            "scientific name",
            "equivalent name",
            "synonym",
            "acronym",
            "blast name",
            "genbank",
            "genbank synonym",
            "genbank acronym",
            "includes",
            "type material",
        ]
        main_field = "scientific name"

        with open(original_file, encoding="utf-8") as f:
            curr_identifier = None
            curr_synonyms = []
            curr_name = None

            for line in f:
                # parse line
                parsed_line = {}
                elements = [e.strip() for e in line.strip().split("|")]
                parsed_line["identifier"] = elements[0]
                parsed_line["name"] = elements[1] if elements[2] == "" else elements[2]
                parsed_line["field"] = elements[3]

                if parsed_line["name"] in ["all", "root"]:
                    continue

                if parsed_line["field"] in ["authority", "in-part", "type material"]:
                    continue

                if parsed_line["field"] not in ncbi_taxonomy_synset:
                    raise ValueError(f"Field {parsed_line['field']} unknown!")

                if curr_identifier is None:
                    curr_identifier = parsed_line["identifier"]

                if curr_identifier == parsed_line["identifier"]:
                    synonym = parsed_line["name"]
                    if parsed_line["field"] == main_field:
                        curr_name = synonym
                    else:
                        curr_synonyms.append(synonym)
                elif curr_identifier != parsed_line["identifier"]:
                    assert curr_name is not None
                    yield Concept(
                        concept_id=curr_identifier,
                        concept_name=curr_name,
                        database_name="NCBI-TAXONOMY",
                    )

                    curr_identifier = parsed_line["identifier"]
                    curr_synonyms = []
                    curr_name = None
                    synonym = parsed_line["name"]
                    if parsed_line["field"] == main_field:
                        curr_name = synonym
                    else:
                        curr_synonyms.append(synonym)
