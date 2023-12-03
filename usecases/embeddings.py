import numpy as np
import rdflib, csv
from sklearn.metrics import pairwise_distances

import os


class EmbeddingRelation(object):
    relation_key: int
    relation_label: str
    fixed_query: str

    @classmethod
    def from_key_and_label_and_fixed_query(cls, key: int, label: str, query: str):
        e = cls()
        e.relation_label = label
        e.relation_key = key
        e.fixed_query = query
        return e

    @classmethod
    def from_key_and_label(cls, key: int, label: str):
        e = cls()
        e.relation_label = label
        e.relation_key = key
        e.fixed_query = None
        return e


class EmbeddingAnswerer(object):
    def __init__(self):
        # Get the absolute path to the current directory
        current_directory = os.path.dirname(os.path.abspath(__file__))

        # Define the relative path to the "data" folder
        data_folder = os.path.join(current_directory, "data")

        # Use absolute paths for loading files from the "data" folder
        entity_emb_path = os.path.join(data_folder, "entity_embeds.npy")
        relation_emb_path = os.path.join(data_folder, "relation_embeds.npy")
        ent_ids_path = os.path.join(data_folder, "entity_ids.del")

        # load the embeddings
        self.entity_emb = np.load(entity_emb_path)
        self.relation_emb = np.load(relation_emb_path)

        with open(ent_ids_path, "r") as ifile:
            self.ent2id = {
                rdflib.term.URIRef(ent): int(idx)
                for idx, ent in csv.reader(ifile, delimiter="\t")
            }
            self.id2ent = {v: k for k, v in self.ent2id.items()}

    @staticmethod
    def is_predicate_in_embedding(relation_label: str) -> EmbeddingRelation | None:
        relation_id = LABELS_IN_RELATION_IDS_DEL.get(relation_label, None)
        if relation_id:
            return EmbeddingRelation.from_key_and_label(relation_id, relation_label)

        return None

    def calculate_embedding_node(
        self, subject, relation_key: int
    ) -> rdflib.IdentifiedNode | None:
        ent_id = self.ent2id.get(subject)
        pred = self.relation_emb[relation_key]
        head = self.entity_emb[ent_id]

        lhs = head + pred
        # compute distance to *any* entity
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        # find most plausible entities
        most_likely = dist.argsort()
        en = self.id2ent[most_likely[0]]

        return en

    def get_n_closest(
        self, nodes: list[rdflib.IdentifiedNode], n=10
    ) -> list[rdflib.IdentifiedNode]:
        if not nodes or not any(nodes):
            return []

        entities = [self.ent2id.get(n, None) for n in nodes]
        if None in entities:
            return []

        center = np.mean(
            [self.entity_emb[e] for e in entities if e is not None], axis=0
        )

        # Retrieve the closest entities
        dist = pairwise_distances(center.reshape(1, -1), self.entity_emb).reshape(-1)
        most_likely_indices = np.argsort(dist)

        closest_entities = [self.id2ent[i] for i in most_likely_indices[:n]]
        return closest_entities


LABELS_IN_RELATION_IDS_DEL = {
    "cast member": 0,
    "notable work": 1,
    "native language": 2,
    "distributed by": 3,
    "occupation": 4,
    "BAMID film rating": 5,
    "subclass of": 6,
    "place of birth": 7,
    "genre": 8,
    "director of photography": 9,
    "country of origin": 10,
    "languages spoken, written or signed": 11,
    "director": 12,
    "instance of": 13,
    "country of citizenship": 14,
    "nominated for": 15,
    "narrative location": 16,
    "sibling": 17,
    "spouse": 18,
    "screenwriter": 19,
    "present in work": 20,
    "creator": 21,
    "award received": 22,
    "filming location": 23,
    "executive producer": 24,
    "diplomatic relation": 25,
    "significant person": 26,
    "production company": 27,
    "country": 28,
    "film editor": 29,
    "different from": 30,
    "set in period": 31,
    "FSK film rating": 32,
    "distribution format": 33,
    "production designer": 34,
    "original language of film or TV show": 35,
    "described by source": 36,
    "based on": 37,
    "main subject": 38,
    "military branch": 39,
    "follows": 40,
    "film crew member": 41,
    "media franchise": 42,
    "place of death": 43,
    "religion": 44,
    "voice actor": 45,
    "color": 46,
    "JMK film rating": 47,
    "educated at": 48,
    "CNC film rating (Romania)": 49,
    "author": 50,
    "twinned administrative body": 52,
    "winner": 53,
    "work location": 54,
    "language of work or name": 55,
    "shares border with": 56,
    "noble title": 57,
    "writing language": 58,
    "member of": 59,
    "followed by": 60,
    "performer": 61,
    "Filmiroda rating": 62,
    "costume designer": 63,
    "IMDA rating": 64,
    "has part": 65,
    "unmarried partner": 66,
    "has quality": 67,
    "student": 68,
    "NMHH film rating": 69,
    "has pet": 70,
    "aspect ratio": 71,
    "assessment": 72,
    "continent": 73,
    "located on street": 74,
    "employer": 75,
    "Medierådet rating": 76,
    "CNC film rating (France)": 77,
    "significant event": 78,
    "from narrative universe": 79,
    "residence": 80,
    "sport": 81,
    "characters": 82,
    "child": 83,
    "located in or next to body of water": 84,
    "contains administrative territorial entity": 85,
    "movement": 86,
    "medical condition": 87,
    "cause of death": 88,
    "country for sport": 89,
    "superhuman feature or ability": 90,
    "part of the series": 91,
    "home world": 92,
    "ethnic group": 93,
    "EIRIN film rating": 94,
    "published in": 95,
    "participant in": 96,
    "RTC film rating": 97,
    "language used": 98,
    "father": 99,
    "contributor to the creative work or subject": 100,
    "RARS rating": 101,
    "headquarters location": 102,
    "IGAC rating": 103,
    "official language": 104,
    "part of": 105,
    "derivative work": 106,
    "BBFC rating": 107,
    "Kijkwijzer rating": 108,
    "takes place in fictional universe": 109,
    "affiliation": 110,
    "interested in": 111,
    "ClassInd rating": 112,
    "capital of": 113,
    "after a work by": 114,
    "animator": 115,
    "ancestral home": 116,
    "relative": 117,
    "copyright license": 118,
    "ICAA rating": 119,
    "location": 120,
    "located in the administrative territorial entity": 121,
    "given name": 122,
    "uses": 123,
    "place of burial": 124,
    "FPB rating": 125,
    "facet of": 126,
    "named after": 127,
    "INCAA film rating": 128,
    "product or material produced": 129,
    "industry": 130,
    "collection": 131,
    "member of the crew of": 132,
    "musical conductor": 133,
    "mother": 134,
    "replaced by": 135,
    "set during recurring event": 136,
    "original broadcaster": 137,
    "has works in the collection": 138,
    "field of work": 139,
    "MPAA film rating": 140,
    "fictional universe described in": 141,
    "Hong Kong film rating": 142,
    "manner of death": 143,
    "conflict": 144,
    "MTRCB rating": 145,
    "developer": 146,
    "location of formation": 147,
    "copyright status": 148,
    "stepparent": 149,
    "sexual orientation": 150,
    "founded by": 151,
    "killed by": 152,
    "historic county": 153,
    "inspired by": 154,
    "first appearance": 155,
    "fabrication method": 156,
    "sound designer": 157,
    "owned by": 158,
    "owner of": 159,
    "platform": 160,
    "storyboard artist": 161,
    "influenced by": 162,
    "IFCO rating": 163,
    "political ideology": 164,
    "replaces": 165,
    "enemy of": 166,
    "permanent resident of": 167,
    "director / manager": 168,
    "student of": 169,
    "RCQ classification": 170,
    "place of publication": 171,
    "allegiance": 172,
    "depicted by": 173,
    "publisher": 174,
    "depicts": 175,
    "partially coincident with": 176,
    "form of creative work": 177,
    "art director": 178,
    "KMRB film rating": 179,
    "participant": 180,
    "presenter": 181,
    "place served by transport hub": 182,
    "time period": 183,
    "quotes work": 184,
    "set in environment": 185,
    "said to be the same as": 186,
    "original film format": 187,
    "crew member(s)": 188,
    "capital": 189,
    "operating system": 190,
    "basin country": 191,
    "choreographer": 192,
    "located on terrain feature": 193,
    "use": 194,
    "partner in business or sport": 195,
    "narrative motif": 196,
    "references work, tradition or theory": 197,
    "character designer": 198,
    "has edition or translation": 199,
    "public holiday": 200,
    "parent organization": 201,
    "convicted of": 202,
    "indigenous to": 203,
    "Australian Classification": 204,
    "opposite of": 205,
    "head of government": 206,
    "make-up artist": 207,
    "conferred by": 208,
    "edition or translation of": 209,
    "located in present-day administrative territorial entity": 210,
    "intended public": 211,
    "input method": 212,
    "member of sports team": 213,
    "cites work": 214,
    "lowest point": 215,
    "designed by": 216,
    "archives at": 217,
    "field of this occupation": 218,
    "presented in": 219,
    "season": 220,
    "broadcast by": 221,
    "applies to jurisdiction": 222,
    "narrator": 223,
    "plot expanded in": 224,
    "operator": 225,
    "has cause": 226,
    "sidekick of": 227,
    "sex or gender": 228,
    "dedicated to": 229,
    "head of state": 230,
    "list of works": 231,
    "operating area": 232,
    "lifestyle": 233,
    "KAVI rating": 234,
    "sports discipline competed in": 235,
    "place of detention": 236,
    "official residence": 237,
    "copyright representative": 238,
    "practiced by": 239,
    "has effect": 240,
    "health specialty": 241,
    "scenographer": 242,
    "represented by": 243,
    "copyright holder": 244,
    "occupant": 245,
    "OFLC classification": 246,
    "business model": 247,
}
