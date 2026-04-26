from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import sqlite3
import re
from typing import Sequence

from .corpus_pipeline import normalize_for_matching
from .embeddings import embed_query_via_http, is_custom_http_embedding_provider
from .indexing import (
    PyViWordSegmenter,
    SparseBM25Encoder,
    build_qdrant_client,
    extract_legal_hint_tokens,
    load_sparse_encoder,
    make_qdrant_point_id,
    require_cross_encoder,
    require_qdrant,
    require_sentence_transformers,
)


ARTICLE_REF_RE = re.compile(r"\bdieu\s+(?P<value>\d+[a-z]?)")
CLAUSE_REF_RE = re.compile(r"\bkhoan\s+(?P<value>\d+)")
POINT_REF_RE = re.compile(r"\bdiem\s+(?P<value>[a-z](?:\.\d+)?)")
YEAR_COUNT_RE = re.compile(r"\b\d+\s+nam\b")

DOCUMENT_KEYWORDS = {
    "45-2019-qh14": (
        "bo luat lao dong",
        "bo luat lao dong 2019",
        "bo luat 2019",
        "45/2019",
        "45/2019/qh14",
        "qh 14",
        "qh14",
    ),
    "nghi-dinh-145-2020-nd-cp": (
        "nghi dinh 145",
        "145/2020",
        "nd-cp",
    ),
}

ACTOR_KEYWORDS = {
    "nguoi_lao_dong": (
        "nguoi lao dong",
        "nhan vien",
        "cong nhan",
        "toi ",
        " em ",
        " minh ",
        "bi cong ty",
        "nghi viec",
        "xin nghi",
    ),
    "nguoi_su_dung_lao_dong": (
        "nguoi su dung lao dong",
        "cong ty",
        "doanh nghiep",
        "chu su dung",
        "nguoi su dung",
    ),
    "lao_dong_nu": (
        "lao dong nu",
        "mang thai",
        "thai san",
        "nuoi con duoi 12 thang",
    ),
    "nguoi_lao_dong_nuoc_ngoai": (
        "lao dong nuoc ngoai",
        "nguoi nuoc ngoai",
        "giay phep lao dong",
    ),
}
GENERIC_ACTOR_FILTERS = frozenset({"nguoi_lao_dong", "nguoi_su_dung_lao_dong"})

TOPIC_KEYWORDS = {
    "cham_dut_hop_dong_lao_dong": (
        "cham dut hop dong",
        "ket thuc hop dong",
        "duoi viec",
        "cho nghi viec",
        "nghi viec",
        "xin nghi",
        "thoi viec",
        "nghi dung quy dinh",
        "nghi dung luat",
        "het han hop dong",
        "cham dut hdld",
    ),
    "don_phuong_cham_dut": (
        "don phuong",
        "tu nghi",
        "nghi ngang",
        "co duoc nghi ngay",
    ),
    "tro_cap": (
        "tro cap",
        "thoi viec",
        "mat viec",
    ),
    "bao_truoc": (
        "bao truoc",
        "thoi han bao truoc",
        "bao truoc bao lau",
    ),
    "ky_luat_sa_thai": (
        "sa thai",
        "ky luat",
        "noi quy lao dong",
    ),
    "thay_doi_co_cau_kinh_te": (
        "thay doi co cau",
        "ly do kinh te",
        "sap nhap",
        "chia tach",
    ),
    "tam_hoan_hop_dong": (
        "tam hoan",
        "tam dung hop dong",
    ),
    "bao_ve_thai_san": (
        "thai san",
        "mang thai",
        "nuoi con duoi 12 thang",
    ),
    "dao_tao_nghe": (
        "dao tao",
        "hoc nghe",
        "chi phi dao tao",
    ),
    "hop_dong_lao_dong": (
        "hop dong lao dong",
        "hdld",
        "giao ket hop dong",
    ),
}

ISSUE_KEYWORDS = {
    "can_cu_cham_dut": (
        "truong hop nao",
        "khi nao duoc",
        "co duoc cham dut",
        "can cu",
        "ly do",
    ),
    "quyen_don_phuong_cham_dut": (
        "don phuong",
        "tu nghi",
        "co duoc nghi ngay",
    ),
    "thoi_han_bao_truoc": (
        "bao truoc",
        "bao truoc bao lau",
        "thoi han bao truoc",
    ),
    "tro_cap_thoi_viec": (
        "tro cap thoi viec",
        "tinh tro cap thoi viec",
    ),
    "tro_cap_mat_viec": (
        "tro cap mat viec",
        "mat viec lam",
    ),
    "nghia_vu_khi_cham_dut": (
        "thanh toan",
        "tra so",
        "phai tra",
        "phai thanh toan",
        "duoc nhan nhung khoan nao",
        "quyen loi con lai",
        "xac nhan thoi gian dong bhxh",
        "nghia vu",
    ),
    "trai_phap_luat": (
        "trai luat",
        "trai phap luat",
        "sai luat",
    ),
    "boi_thuong": (
        "boi thuong",
        "den bu",
    ),
    "sa_thai": (
        "sa thai",
    ),
    "noi_quy_lao_dong": (
        "noi quy",
        "ky luat",
    ),
    "thong_bao_cham_dut": (
        "thong bao",
        "bao truoc",
    ),
}
DEFAULT_MAX_CONTEXT_CHARS = 8000
DEFAULT_MAX_CONTEXT_TOKENS = 1400
CALCULATION_QUERY_HINTS = (
    "cach tinh",
    "duoc tinh",
    "tinh nhu the nao",
    "tinh the nao",
    "bao nhieu",
)
CALCULATION_CONTEXT_HINTS = (
    "de tinh",
    "moi nam",
    "tong thoi gian",
    "tien luong",
    "binh quan",
    "mot nua thang",
)
IMPLEMENTATION_DETAIL_HINTS = (
    "chi tiet",
    "huong dan",
    "nghi dinh",
    "chinh phu",
)
DELEGATION_CONTEXT_HINTS = (
    "chinh phu quy dinh chi tiet dieu nay",
    "quy dinh chi tiet dieu nay",
)
TERMINATION_QUERY_HINTS = (
    "cham dut hop dong",
    "nghi viec",
    "xin nghi",
    "thoi viec",
    "het han hop dong",
    "nghi dung quy dinh",
    "nghi dung luat",
)
TERMINATION_SECTION_HINTS = ("cham dut hop dong lao dong",)
TERMINATION_BENEFIT_CONTEXT_HINTS = (
    "tro cap thoi viec",
    "tro cap mat viec",
    "bao hiem that nghiep",
    "mat viec lam",
)
BENEFIT_COMPUTATION_QUERY_HINTS = (
    "bao hiem that nghiep",
    "da dong bao hiem that nghiep",
    "lam o cong ty",
)
MATERNITY_CONTEXT_HINTS = (
    "thai san",
    "mang thai",
    "nuoi con duoi 12 thang",
)
RETIREMENT_CONTEXT_HINTS = (
    "nghi huu",
    "luong huu",
)
TOKEN_ESTIMATE_RE = re.compile(r"\S+")
DEFAULT_RERANKER_TOP_N = 24
RECORD_SOURCE_SQLITE = "sqlite"
RECORD_SOURCE_QDRANT_PAYLOAD = "qdrant_payload"
SUPPORTED_RECORD_SOURCES = {RECORD_SOURCE_SQLITE, RECORD_SOURCE_QDRANT_PAYLOAD}
RRF_K = 60.0


@dataclass(frozen=True)
class RuleBasedQueryExpansion:
    phrases: tuple[str, ...]
    articles: tuple[str, ...]
    topics: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()
    expansions: tuple[str, ...] = ()
    excluded_phrases: tuple[str, ...] = ()


TERMINATION_ARTICLE_MAP = {
    "34": (
        "can cu cham dut hop dong",
        "cac truong hop cham dut hop dong",
        "het han hop dong",
        "thoa thuan cham dut",
        "giay phep lao dong het hieu luc",
    ),
    "35": (
        "nguoi lao dong don phuong",
        "xin nghi",
        "bao truoc khi nghi viec",
        "nghi viec khong can bao truoc",
        "nghi ngang",
        "bi no luong",
        "cham luong",
    ),
    "36": (
        "cong ty don phuong",
        "nguoi su dung lao dong don phuong",
        "thuong xuyen khong hoan thanh cong viec",
        "om dau keo dai",
        "hoa hoan",
        "thien tai",
        "tu y bo viec",
        "tu y nghi viec",
        "cung cap thong tin khong trung thuc",
    ),
    "37": (
        "khong duoc don phuong cham dut",
        "dang nghi om",
        "nghi hang nam",
        "nghi phep nam",
        "nghi viec rieng",
        "dang mang thai",
        "nghi thai san",
    ),
    "38": (
        "rut don nghi viec",
        "doi y nghi viec",
        "huy bo viec don phuong",
        "huy quyet dinh don phuong",
    ),
    "39": (
        "don phuong cham dut trai phap luat la gi",
        "the nao la don phuong cham dut trai phap luat",
        "nghi ngang trai luat",
    ),
    "40": (
        "nghia vu cua nguoi lao dong khi don phuong trai phap luat",
        "nghi ngang phai boi thuong",
        "khong bao truoc phai boi thuong",
        "hoan tra chi phi dao tao",
    ),
    "41": (
        "nghia vu cua nguoi su dung lao dong khi don phuong trai phap luat",
        "cong ty cham dut trai luat",
        "sa thai trai luat",
        "boi thuong it nhat 02 thang tien luong",
        "nhan nguoi lao dong tro lai lam viec",
    ),
    "46": (
        "tro cap thoi viec",
        "tinh tro cap thoi viec",
        "thoi gian lam viec de tinh tro cap",
        "tien luong tinh tro cap thoi viec",
    ),
    "47": (
        "tro cap mat viec",
        "mat viec lam",
        "thay doi co cau",
        "thay doi cong nghe",
        "ly do kinh te",
    ),
    "48": (
        "trach nhiem khi cham dut hop dong",
        "thanh toan khi nghi viec",
        "thanh toan tien luong",
        "chot so bhxh",
        "tra so bhxh",
        "giam so bhxh",
        "tra lai giay to",
        "14 ngay",
        "30 ngay",
    ),
    "122": (
        "xu ly ky luat lao dong",
        "hop xu ly ky luat",
        "nguyen tac xu ly ky luat",
        "khong duoc xu ly ky luat",
        "mang thai xu ly ky luat",
    ),
    "124": (
        "hinh thuc xu ly ky luat",
        "khien trach",
        "keo dai thoi han nang luong",
        "cach chuc",
        "sa thai la hinh thuc ky luat",
    ),
    "125": (
        "ap dung hinh thuc sa thai",
        "ky luat sa thai",
        "bi sa thai",
        "tu y bo viec",
        "tu y nghi viec",
        "trom cap",
        "tham o",
        "danh bac",
        "su dung ma tuy",
    ),
    "128": (
        "tam dinh chi cong viec",
        "tam ung tien luong khi bi tam dinh chi",
        "xac minh vu viec vi pham ky luat",
    ),
    "129": (
        "boi thuong thiet hai",
        "lam hu hong dung cu thiet bi",
        "lam xuoc laptop",
        "lam mat tai san",
        "khau tru luong de boi thuong",
    ),
}

TERMINATION_ARTICLE_TOPIC_HINTS = {
    **{
        article: ("cham_dut_hop_dong_lao_dong",)
        for article in ("34", "35", "36", "37", "38", "39", "40", "41", "46", "47", "48")
    },
    **{article: ("ky_luat_sa_thai",) for article in ("122", "124", "125", "128")},
}
TERMINATION_ARTICLE_ISSUE_HINTS = {
    "35": ("quyen_don_phuong_cham_dut",),
    "41": ("boi_thuong",),
    "46": ("tro_cap_thoi_viec",),
    "47": ("tro_cap_mat_viec",),
    "48": ("nghia_vu_khi_cham_dut",),
    "125": ("sa_thai",),
    "129": ("boi_thuong",),
}
TERMINATION_ARTICLE_EXCLUDED_HINTS = {
    "35": ("nghi viec rieng",),
}

TERMINATION_ARTICLE_QUERY_RULES = tuple(
    RuleBasedQueryExpansion(
        phrases=phrases,
        articles=(article,),
        topics=TERMINATION_ARTICLE_TOPIC_HINTS.get(article, ()),
        issues=TERMINATION_ARTICLE_ISSUE_HINTS.get(article, ()),
        expansions=(f"Dieu {article}",),
        excluded_phrases=TERMINATION_ARTICLE_EXCLUDED_HINTS.get(article, ()),
    )
    for article, phrases in TERMINATION_ARTICLE_MAP.items()
)

RETRIEVAL_MISS_QUERY_RULES = (
    RuleBasedQueryExpansion(
        phrases=("luong cham", "cham luong", "no luong", "tra luong tre", "tra luong khong dung han"),
        articles=("97", "35"),
        topics=("don_phuong_cham_dut",),
        issues=("quyen_don_phuong_cham_dut",),
        expansions=(
            "khong duoc tra du luong hoac tra luong khong dung thoi han",
            "khoan 4 Dieu 97 tra luong cham",
            "diem b khoan 2 Dieu 35 khong can bao truoc",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("nghi phep nam", "phep nam", "nghi hang nam", "ngay phep chua nghi"),
        articles=("113", "114", "37"),
        issues=("nghia_vu_khi_cham_dut",),
        expansions=(
            "nghi hang nam",
            "thanh toan tien luong nhung ngay chua nghi",
            "nguoi lao dong dang nghi hang nam",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("nghi viec rieng", "viec rieng", "nghi rieng"),
        articles=("115", "37"),
        expansions=(
            "nghi viec rieng ma van huong nguyen luong",
            "nguoi lao dong dang nghi viec rieng",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("giam so bhxh", "giu so bhxh", "khong tra so bhxh", "chot so bhxh", "so bao hiem xa hoi"),
        articles=("48", "17"),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("nghia_vu_khi_cham_dut",),
        expansions=(
            "hoan thanh thu tuc xac nhan thoi gian dong bao hiem xa hoi",
            "tra lai ban chinh giay to neu da giu cua nguoi lao dong",
            "giu ban chinh giay to cua nguoi lao dong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("giay uy quyen", "uy quyen ky", "uy quyen cham dut", "tham quyen ky"),
        articles=("18", "45"),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong", "thong_bao_cham_dut"),
        expansions=(
            "nguoi dai dien theo phap luat hoac nguoi duoc uy quyen",
            "tham quyen giao ket hop dong lao dong",
            "thong bao bang van ban ve viec cham dut hop dong lao dong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("khoi kien", "toa an", "toa an nhan dan", "thoi hieu yeu cau toa an"),
        articles=("188", "190"),
        expansions=(
            "tranh chap lao dong ca nhan",
            "hoa giai vien lao dong",
            "thoi hieu yeu cau toa an giai quyet tranh chap lao dong ca nhan",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("duoi viec", "cho nghi viec", "cho thoi viec"),
        articles=("36", "41", "124", "125"),
        topics=("cham_dut_hop_dong_lao_dong", "ky_luat_sa_thai"),
        issues=("can_cu_cham_dut", "boi_thuong", "sa_thai"),
        expansions=(
            "nguoi su dung lao dong don phuong cham dut hop dong lao dong",
            "nghia vu cua nguoi su dung lao dong khi don phuong cham dut trai phap luat",
            "ap dung hinh thuc xu ly ky luat sa thai",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("nghi ngang", "bo viec", "tu y nghi viec", "tu y bo viec"),
        articles=("35", "36", "39", "40", "125"),
        topics=("don_phuong_cham_dut", "ky_luat_sa_thai"),
        issues=("quyen_don_phuong_cham_dut", "trai_phap_luat", "sa_thai"),
        expansions=(
            "nguoi lao dong don phuong cham dut hop dong lao dong trai phap luat",
            "khong bao truoc",
            "tu y bo viec 05 ngay cong don trong thoi han 30 ngay",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("ep viet don nghi", "ep viet don xin nghi", "buoc viet don nghi", "bat viet don nghi"),
        articles=("34", "35", "36", "41"),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("trai_phap_luat", "boi_thuong"),
        expansions=(
            "thoa thuan cham dut hop dong lao dong",
            "don phuong cham dut hop dong lao dong trai phap luat",
            "nguoi su dung lao dong cham dut hop dong lao dong trai phap luat",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("luong thang 13", "thang luong 13", "thuong tet", "tien thuong"),
        articles=("104",),
        expansions=(
            "thuong",
            "tien thuong",
            "quy che thuong do nguoi su dung lao dong quyet dinh",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("can tru phep", "tru phep nam", "khong thanh toan phep", "ngay phep chua nghi"),
        articles=("113", "114", "48"),
        issues=("nghia_vu_khi_cham_dut",),
        expansions=(
            "nghi hang nam",
            "thanh toan tien luong cho nhung ngay chua nghi",
            "trach nhiem khi cham dut hop dong lao dong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("bat nghi khong luong", "nghi khong luong", "nghi khong huong luong", "ngung viec"),
        articles=("99", "115"),
        expansions=(
            "nghi khong huong luong",
            "tien luong ngung viec",
            "nguoi lao dong ngung viec",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "chuyen viec khac",
            "chuyen lam viec khac",
            "dieu chuyen cong viec",
            "lam cong viec khac",
            "lam viec khac so voi hop dong",
        ),
        articles=("29",),
        topics=("hop_dong_lao_dong",),
        expansions=(
            "chuyen nguoi lao dong lam cong viec khac so voi hop dong lao dong",
            "bao truoc it nhat 03 ngay lam viec",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("khong hoan thanh cong viec", "thuong xuyen khong hoan thanh", "khong dat kpi"),
        articles=("36",),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("can_cu_cham_dut",),
        expansions=(
            "thuong xuyen khong hoan thanh cong viec theo hop dong lao dong",
            "quy che danh gia muc do hoan thanh cong viec",
        ),
    ),
)

LEGAL_ISSUE_ARTICLE_MAP = {
    "can_cu_cham_dut": ("34", "36"),
    "quyen_don_phuong_cham_dut": ("35",),
    "thoi_han_bao_truoc": ("35", "36", "37"),
    "tro_cap_thoi_viec": ("46",),
    "tro_cap_mat_viec": ("47",),
    "nghia_vu_khi_cham_dut": ("48",),
    "trai_phap_luat": ("39", "40", "41"),
    "boi_thuong": ("40", "41", "129"),
    "sa_thai": ("124", "125"),
    "noi_quy_lao_dong": ("122", "124", "125"),
    "thong_bao_cham_dut": ("35", "45"),
}

LEGAL_TOPIC_ARTICLE_MAP = {
    "tro_cap": ("46", "47"),
    "bao_truoc": ("35", "36", "37"),
    "ky_luat_sa_thai": ("122", "124", "125"),
    "thay_doi_co_cau_kinh_te": ("42", "44", "47"),
    "tam_hoan_hop_dong": ("30", "31"),
    "bao_ve_thai_san": ("137", "138"),
}

LEGAL_ISSUE_QUERY_HINTS = {
    "can_cu_cham_dut": ("cac truong hop cham dut hop dong lao dong",),
    "quyen_don_phuong_cham_dut": ("quyen don phuong cham dut hop dong lao dong cua nguoi lao dong",),
    "thoi_han_bao_truoc": ("thoi han bao truoc khi don phuong cham dut hop dong lao dong",),
    "tro_cap_thoi_viec": ("tro cap thoi viec moi nam lam viec nua thang tien luong",),
    "tro_cap_mat_viec": ("tro cap mat viec it nhat bang 02 thang tien luong",),
    "nghia_vu_khi_cham_dut": ("thanh toan xac nhan thoi gian dong bao hiem xa hoi tra lai giay to",),
    "trai_phap_luat": ("don phuong cham dut hop dong lao dong trai phap luat",),
    "boi_thuong": ("boi thuong khi cham dut hop dong lao dong trai phap luat",),
    "sa_thai": ("ap dung hinh thuc xu ly ky luat sa thai",),
    "noi_quy_lao_dong": ("nguyen tac trinh tu xu ly ky luat lao dong",),
    "thong_bao_cham_dut": ("thong bao bang van ban ve viec cham dut hop dong lao dong",),
}

QUERY_TYPE_KEYWORDS = {
    "yes_no": ("co duoc", "co phai", "dung luat khong", "sai luat khong", "duoc khong"),
    "time_limit": ("bao lau", "bao nhieu ngay", "thoi han", "toi da", "it nhat", "ngay lam viec"),
    "money_percentage": (
        "bao nhieu tien",
        "bao nhieu phan tram",
        "phan tram",
        "muc luong",
        "tien luong",
        "lai",
        "boi thuong",
        "tro cap",
    ),
    "procedure": ("thu tuc", "quy trinh", "ho so", "phai lam gi", "thanh phan tham gia", "bien ban"),
    "remedy": ("boi thuong", "den bu", "nhan lai lam viec", "khoi phuc", "yeu cau cong ty"),
    "definition": ("la gi", "the nao la", "duoc hieu la", "dinh nghia"),
    "classification": ("co phai la", "khac gi", "thuoc truong hop nao", "truong hop nao"),
    "missing_fact": ("can biet", "co can", "neu", "truong hop"),
}


@dataclass(frozen=True)
class QueryIntent:
    raw_query: str
    normalized_query: str
    actor_filters: tuple[str, ...]
    topic_filters: tuple[str, ...]
    issue_filters: tuple[str, ...]
    document_filters: tuple[str, ...]
    article_numbers: tuple[str, ...] = ()
    inferred_article_numbers: tuple[str, ...] = ()
    clause_refs: tuple[str, ...] = ()
    point_refs: tuple[str, ...] = ()
    query_expansions: tuple[str, ...] = ()
    query_types: tuple[str, ...] = ()

    @property
    def all_article_numbers(self) -> tuple[str, ...]:
        return dedupe_preserve_order((*self.article_numbers, *self.inferred_article_numbers))

    @property
    def article_number(self) -> str | None:
        article_numbers = self.all_article_numbers
        return article_numbers[0] if article_numbers else None

    @property
    def clause_ref(self) -> str | None:
        return self.clause_refs[0] if self.clause_refs else None

    @property
    def point_ref(self) -> str | None:
        return self.point_refs[0] if self.point_refs else None

    @property
    def legal_reference_filters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        filters: list[tuple[str, tuple[str, ...]]] = []
        article_numbers = self.all_article_numbers
        if article_numbers:
            filters.append(("article_number", article_numbers))
        if self.clause_refs:
            filters.append(("clause_ref", self.clause_refs))
        if self.point_refs:
            filters.append(("point_ref", self.point_refs))
        return tuple(filters)

    @property
    def explicit_legal_reference_filters(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        filters: list[tuple[str, tuple[str, ...]]] = []
        if self.article_numbers:
            filters.append(("article_number", self.article_numbers))
        if self.clause_refs:
            filters.append(("clause_ref", self.clause_refs))
        if self.point_refs:
            filters.append(("point_ref", self.point_refs))
        return tuple(filters)


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    qdrant_point_id: str
    score: float
    citation_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievedRecord:
    chunk_id: str
    parent_chunk_id: str | None
    citation_text: str
    text: str
    dense_text: str
    sparse_text: str
    payload: dict[str, object]


@dataclass(frozen=True)
class RetrievalContext:
    chunk_id: str
    citation_text: str
    text: str
    payload: dict[str, object]
    score: float
    matched_chunk_ids: tuple[str, ...]
    matched_citations: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalResult:
    query: str
    intent: QueryIntent
    hits: tuple[SearchHit, ...]
    contexts: tuple[RetrievalContext, ...]


def resolve_record_source(manifest: dict[str, object]) -> str:
    configured = (
        os.getenv("RETRIEVER_RECORD_SOURCE", "").strip()
        or str(manifest.get("record_source") or "").strip()
        or RECORD_SOURCE_SQLITE
    )
    if configured not in SUPPORTED_RECORD_SOURCES:
        raise ValueError(
            "RETRIEVER_RECORD_SOURCE must be one of: "
            + ", ".join(sorted(SUPPORTED_RECORD_SOURCES))
        )
    return configured


def record_from_qdrant_payload(payload: dict[str, object]) -> RetrievedRecord:
    chunk_id = str(payload.get("chunk_id") or "").strip()
    if not chunk_id:
        raise ValueError("Qdrant payload is missing chunk_id.")

    return RetrievedRecord(
        chunk_id=chunk_id,
        parent_chunk_id=(
            str(payload.get("parent_chunk_id"))
            if payload.get("parent_chunk_id")
            else None
        ),
        citation_text=str(payload.get("citation_text") or ""),
        text=str(payload.get("text") or ""),
        dense_text=str(payload.get("dense_text") or ""),
        sparse_text=str(payload.get("sparse_text") or ""),
        payload=dict(payload),
    )


def record_reference_sort_key(record: RetrievedRecord) -> tuple[int, int, str, str, str]:
    level_order = {"article": 0, "clause": 1, "point": 2}
    level = str(record.payload.get("level") or "")
    clause_ref = str(record.payload.get("clause_ref") or "")
    point_ref = str(record.payload.get("point_ref") or "")
    try:
        clause_order = int(clause_ref)
    except ValueError:
        clause_order = 10_000
    return (
        level_order.get(level, 3),
        clause_order,
        clause_ref,
        point_ref,
        record.chunk_id,
    )


def dedupe_preserve_order(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return tuple(ordered)


def parse_reference_values(pattern: re.Pattern[str], normalized_query: str) -> tuple[str, ...]:
    return dedupe_preserve_order(
        tuple(match.group("value").lower() for match in pattern.finditer(normalized_query))
    )


def collect_keyword_matches(normalized_query: str, mapping: dict[str, Sequence[str]]) -> tuple[str, ...]:
    matches = [
        label
        for label, keywords in mapping.items()
        if any(keyword in normalized_query for keyword in keywords)
    ]
    return tuple(matches)


def contains_normalized_phrase(normalized_text: str, phrases: Sequence[str]) -> bool:
    return any(phrase in normalized_text for phrase in phrases)


def collect_rule_based_query_expansions(
    normalized_query: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    inferred_articles: list[str] = []
    topics: list[str] = []
    issues: list[str] = []
    expansions: list[str] = []

    for rule in (*RETRIEVAL_MISS_QUERY_RULES, *TERMINATION_ARTICLE_QUERY_RULES):
        if not contains_normalized_phrase(normalized_query, rule.phrases):
            continue
        if rule.excluded_phrases and contains_normalized_phrase(
            normalized_query,
            rule.excluded_phrases,
        ):
            continue
        inferred_articles.extend(rule.articles)
        topics.extend(rule.topics)
        issues.extend(rule.issues)
        expansions.extend(rule.expansions)

    return (
        dedupe_preserve_order(inferred_articles),
        dedupe_preserve_order(topics),
        dedupe_preserve_order(issues),
        dedupe_preserve_order(expansions),
    )


def collect_mapped_article_expansions(
    *,
    topic_filters: Sequence[str],
    issue_filters: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    articles: list[str] = []
    expansions: list[str] = []
    for issue in issue_filters:
        articles.extend(LEGAL_ISSUE_ARTICLE_MAP.get(issue, ()))
        expansions.extend(LEGAL_ISSUE_QUERY_HINTS.get(issue, ()))
    for topic in topic_filters:
        articles.extend(LEGAL_TOPIC_ARTICLE_MAP.get(topic, ()))
    expansions.extend(f"Dieu {article}" for article in articles)
    return dedupe_preserve_order(articles), dedupe_preserve_order(expansions)


def filter_specific_actor_labels(actor_labels: Sequence[str]) -> tuple[str, ...]:
    return tuple(label for label in actor_labels if label not in GENERIC_ACTOR_FILTERS)


def prioritize_issue_filters(issue_labels: Sequence[str]) -> tuple[str, ...]:
    prioritized = tuple(issue_labels)
    if any(label in prioritized for label in ("tro_cap_thoi_viec", "tro_cap_mat_viec")):
        return tuple(label for label in prioritized if label != "nghia_vu_khi_cham_dut")
    return prioritized


def route_query(query: str) -> QueryIntent:
    normalized_query = normalize_for_matching(f" {query} ")
    inferred_articles, inferred_topics, inferred_issues, query_expansions = (
        collect_rule_based_query_expansions(normalized_query)
    )
    topic_filters = dedupe_preserve_order(
        (*collect_keyword_matches(normalized_query, TOPIC_KEYWORDS), *inferred_topics)
    )
    issue_filters = dedupe_preserve_order(
        (*collect_keyword_matches(normalized_query, ISSUE_KEYWORDS), *inferred_issues)
    )
    mapped_articles, mapped_expansions = collect_mapped_article_expansions(
        topic_filters=topic_filters,
        issue_filters=issue_filters,
    )
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=collect_keyword_matches(normalized_query, ACTOR_KEYWORDS),
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        document_filters=collect_keyword_matches(normalized_query, DOCUMENT_KEYWORDS),
        article_numbers=parse_reference_values(ARTICLE_REF_RE, normalized_query),
        inferred_article_numbers=dedupe_preserve_order((*inferred_articles, *mapped_articles)),
        clause_refs=parse_reference_values(CLAUSE_REF_RE, normalized_query),
        point_refs=parse_reference_values(POINT_REF_RE, normalized_query),
        query_expansions=dedupe_preserve_order((*query_expansions, *mapped_expansions)),
        query_types=collect_keyword_matches(normalized_query, QUERY_TYPE_KEYWORDS),
    )


def load_manifest(index_path: Path) -> dict[str, object]:
    if index_path.is_dir():
        manifest_path = index_path / "current.json"
    else:
        manifest_path = index_path
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def format_intent_summary(intent: QueryIntent) -> str:
    parts: list[str] = []
    if intent.document_filters:
        parts.append(f"document={', '.join(intent.document_filters)}")
    if intent.actor_filters:
        parts.append(f"actor={', '.join(intent.actor_filters)}")
    if intent.topic_filters:
        parts.append(f"topic={', '.join(intent.topic_filters)}")
    if intent.issue_filters:
        parts.append(f"issue={', '.join(intent.issue_filters)}")
    if intent.article_numbers:
        parts.append(f"dieu={', '.join(intent.article_numbers)}")
    if intent.inferred_article_numbers:
        parts.append(f"dieu_suy_luan={', '.join(intent.inferred_article_numbers)}")
    if intent.clause_refs:
        parts.append(f"khoan={', '.join(intent.clause_refs)}")
    if intent.point_refs:
        parts.append(f"diem={', '.join(intent.point_refs)}")
    if intent.query_types:
        parts.append(f"loai_cau_hoi={', '.join(intent.query_types)}")
    if intent.query_expansions:
        parts.append(f"mo_rong={'; '.join(intent.query_expansions)}")
    return "; ".join(parts) if parts else "khong co filter heuristic"


def build_query_variants(intent: QueryIntent) -> tuple[str, ...]:
    variants: list[str] = [intent.raw_query.strip()]

    if intent.query_expansions:
        variants.append(" ".join((intent.raw_query, *intent.query_expansions)).strip())

    issue_parts: list[str] = []
    for issue in intent.issue_filters:
        issue_parts.extend(LEGAL_ISSUE_QUERY_HINTS.get(issue, ()))
    if issue_parts:
        variants.append(" ".join(dedupe_preserve_order(issue_parts)))

    if intent.all_article_numbers:
        reference_suffix = " ".join(
            (
                *(f"khoan {clause}" for clause in intent.clause_refs),
                *(f"diem {point}" for point in intent.point_refs),
            )
        )
        citation_parts = [
            " ".join(part for part in [f"Dieu {article}", reference_suffix, "Bo luat Lao dong 2019"] if part)
            for article in intent.all_article_numbers
        ]
        citation_parts.extend(intent.query_expansions[:3])
        variants.append(" ".join(citation_parts))

    return tuple(
        variant
        for variant in dedupe_preserve_order(tuple(variant for variant in variants if variant))
        if variant
    )


def build_context_block(context: RetrievalContext, index: int) -> str:
    lines = [
        f"[NGU CANH {index}]",
        f"Co so phap ly: {context.citation_text}",
    ]

    unique_matched_citations = dedupe_preserve_order(context.matched_citations)
    if unique_matched_citations and unique_matched_citations != (context.citation_text,):
        lines.append("Match goc:")
        lines.extend(f"- {citation}" for citation in unique_matched_citations)

    lines.extend(
        [
            "Noi dung:",
            context.text.strip(),
        ]
    )
    return "\n".join(lines).strip()


def estimate_token_count(text: str) -> int:
    return len(TOKEN_ESTIMATE_RE.findall(text))


def context_article_key(context: RetrievalContext) -> tuple[str, str] | None:
    document_id = str(context.payload.get("document_id") or "")
    article_number = str(context.payload.get("article_number") or "")
    if not document_id or not article_number:
        return None
    return document_id, article_number


def diversify_contexts_by_article(contexts: Sequence[RetrievalContext]) -> tuple[RetrievalContext, ...]:
    first_per_article: list[RetrievalContext] = []
    remaining: list[RetrievalContext] = []
    seen_article_keys: set[tuple[str, str]] = set()

    for context in contexts:
        key = context_article_key(context)
        if key is None:
            first_per_article.append(context)
            continue
        if key not in seen_article_keys:
            seen_article_keys.add(key)
            first_per_article.append(context)
        else:
            remaining.append(context)

    return tuple((*first_per_article, *remaining))


def select_contexts_for_prompt(
    contexts: Sequence[RetrievalContext],
    *,
    max_contexts: int | None = None,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> tuple[RetrievalContext, ...]:
    ranked_contexts = diversify_contexts_by_article(contexts)
    limited_contexts = ranked_contexts[:max_contexts] if max_contexts is not None else ranked_contexts
    selected: list[RetrievalContext] = []
    current_len = 0
    current_tokens = 0

    for context in limited_contexts:
        block = build_context_block(context, len(selected) + 1)
        block_tokens = estimate_token_count(block)
        separator_len = 2 if selected else 0
        next_len = current_len + separator_len + len(block)
        next_tokens = current_tokens + block_tokens
        exceeds_char_budget = max_chars > 0 and next_len > max_chars
        exceeds_token_budget = max_tokens is not None and max_tokens > 0 and next_tokens > max_tokens

        if selected and (exceeds_char_budget or exceeds_token_budget):
            break

        if not exceeds_char_budget and not exceeds_token_budget:
            selected.append(context)
            current_len = next_len
            current_tokens = next_tokens
            continue

        # Preserve the highest-ranked block intact instead of truncating legal text mid-sentence.
        if not selected:
            selected.append(context)
            current_len = next_len
            current_tokens = next_tokens
        break

    return tuple(selected)


def format_context_for_prompt(
    contexts: Sequence[RetrievalContext],
    *,
    max_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    max_tokens: int | None = DEFAULT_MAX_CONTEXT_TOKENS,
) -> str:
    selected_contexts = select_contexts_for_prompt(
        contexts,
        max_chars=max_chars,
        max_tokens=max_tokens,
    )
    blocks = [
        build_context_block(context, index)
        for index, context in enumerate(selected_contexts, start=1)
    ]

    return "\n\n".join(blocks).strip()


class HybridRetriever:
    def __init__(
        self,
        *,
        index_path: Path = Path("artifacts/index"),
        device: str | None = None,
        reranker_model: str | None = None,
        reranker_top_n: int = DEFAULT_RERANKER_TOP_N,
    ) -> None:
        self._manifest = load_manifest(index_path)
        self._device = device or str(self._manifest.get("device") or "cpu")
        self._record_source = resolve_record_source(self._manifest)
        records_db_path = str(self._manifest.get("records_db_path") or "").strip()
        qdrant_path = str(self._manifest.get("qdrant_path") or "").strip()
        self._records_db_path = Path(records_db_path) if records_db_path else None
        self._qdrant_path = Path(qdrant_path) if qdrant_path else None
        self._collection_name = (
            os.getenv("QDRANT_COLLECTION", "").strip()
            or str(self._manifest["collection_name"])
        )
        self._dense_model_name = str(self._manifest["dense_model_name"])
        self._dense_vector_name = str(self._manifest["dense_vector_name"])
        self._sparse_vector_name = str(self._manifest["sparse_vector_name"])
        self._segmenter = PyViWordSegmenter()
        self._sparse_encoder = load_sparse_encoder(Path(str(self._manifest["sparse_encoder_path"])))
        self._sqlite: sqlite3.Connection | None = None
        if self._record_source == RECORD_SOURCE_SQLITE:
            if self._records_db_path is None:
                raise ValueError("records_db_path is required when record_source is sqlite.")
            self._sqlite = sqlite3.connect(self._records_db_path)
            self._sqlite.row_factory = sqlite3.Row
        qdrant_client_cls, self._qdrant_models = require_qdrant()
        self._qdrant = build_qdrant_client(qdrant_client_cls, self._qdrant_path)
        self._dense_model = None
        self._reranker_model_name = str(reranker_model or "").strip()
        self._reranker_top_n = max(1, int(reranker_top_n))
        self._reranker = None

    @property
    def manifest(self) -> dict[str, object]:
        return dict(self._manifest)

    @property
    def reranker_model_name(self) -> str:
        return self._reranker_model_name

    @property
    def reranker_enabled(self) -> bool:
        return bool(self._reranker_model_name)

    def close(self) -> None:
        self._qdrant.close()
        if self._sqlite is not None:
            self._sqlite.close()

    def _get_dense_model(self):
        if self._dense_model is None:
            sentence_transformer_cls = require_sentence_transformers()
            self._dense_model = sentence_transformer_cls(self._dense_model_name, device=self._device)
        return self._dense_model

    def _get_reranker(self):
        if not self.reranker_enabled:
            return None
        if self._reranker is None:
            cross_encoder_cls = require_cross_encoder()
            self._reranker = cross_encoder_cls(self._reranker_model_name, device=self._device)
        return self._reranker

    def _encode_dense_query(self, query: str) -> list[float]:
        if is_custom_http_embedding_provider():
            return embed_query_via_http(query)

        model = self._get_dense_model()
        vector = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    def _encode_sparse_query(
        self,
        intent: QueryIntent,
        query_text: str | None = None,
    ) -> tuple[list[str], object]:
        sparse_query_text = query_text or "\n".join(
            part for part in (intent.raw_query, *intent.query_expansions) if part
        )
        tokens = self._segmenter.segment(sparse_query_text)
        tokens.extend(extract_legal_hint_tokens(sparse_query_text))
        tokens.extend(f"dieu_{value}" for value in intent.all_article_numbers)
        tokens.extend(f"khoan_{value}" for value in intent.clause_refs)
        tokens.extend(f"diem_{value}" for value in intent.point_refs)
        tokens.extend(f"topic_{value}" for value in intent.topic_filters)
        tokens.extend(f"issue_{value}" for value in intent.issue_filters)
        tokens.extend(f"actor_{value}" for value in filter_specific_actor_labels(intent.actor_filters))
        tokens.extend(f"qtype_{value}" for value in intent.query_types)
        sparse_query = self._sparse_encoder.encode_query(tokens)
        sparse_vector = self._qdrant_models.SparseVector(
            indices=sparse_query.indices,
            values=sparse_query.values,
        )
        return tokens, sparse_vector

    def _build_query_filter(self, intent: QueryIntent):
        must_conditions: list[object] = []
        models = self._qdrant_models

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        for field_name, values in intent.explicit_legal_reference_filters:
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(values)),
                )
            )

        return models.Filter(must=must_conditions) if must_conditions else None

    def _build_reference_boost_filter(self, intent: QueryIntent):
        if not intent.legal_reference_filters:
            return None

        models = self._qdrant_models
        must_conditions: list[object] = []

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        for field_name, values in intent.legal_reference_filters:
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(values)),
                )
            )

        return models.Filter(must=must_conditions)

    def _build_issue_focus_filter(self, intent: QueryIntent):
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        if not prioritized_issue_filters:
            return None

        models = self._qdrant_models
        must_conditions: list[object] = []

        if intent.document_filters:
            must_conditions.append(
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchAny(any=list(intent.document_filters)),
                )
            )

        must_conditions.append(
            models.FieldCondition(
                key="issue_type",
                match=models.MatchAny(any=list(prioritized_issue_filters)),
            )
        )
        return models.Filter(must=must_conditions)

    def _records_from_rows(self, rows: Sequence[sqlite3.Row]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for row in rows:
            payload = json.loads(row["payload_json"])
            records[str(row["chunk_id"])] = RetrievedRecord(
                chunk_id=str(row["chunk_id"]),
                parent_chunk_id=str(row["parent_chunk_id"]) if row["parent_chunk_id"] else None,
                citation_text=str(row["citation_text"]),
                text=str(row["text"]),
                dense_text=str(row["dense_text"]),
                sparse_text=str(row["sparse_text"]),
                payload=payload,
            )
        return records

    def _uses_qdrant_payload_records(self) -> bool:
        return (
            getattr(self, "_record_source", RECORD_SOURCE_SQLITE)
            == RECORD_SOURCE_QDRANT_PAYLOAD
        )

    def _records_from_qdrant_points(self, points: Sequence[object]) -> dict[str, RetrievedRecord]:
        records: dict[str, RetrievedRecord] = {}
        for point in points:
            payload = getattr(point, "payload", None)
            if not isinstance(payload, dict):
                continue
            try:
                record = record_from_qdrant_payload(payload)
            except ValueError:
                continue
            records[record.chunk_id] = record
        return records

    def _fetch_records_from_qdrant_ids(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        points = self._qdrant.retrieve(
            collection_name=self._collection_name,
            ids=[make_qdrant_point_id(chunk_id) for chunk_id in ordered_ids],
            with_payload=True,
            with_vectors=False,
        )
        return self._records_from_qdrant_points(points)

    def _fetch_records_from_hits(self, hits: Sequence[SearchHit]) -> dict[str, RetrievedRecord]:
        if not self._uses_qdrant_payload_records():
            return self._fetch_records([hit.chunk_id for hit in hits])

        records: dict[str, RetrievedRecord] = {}
        missing_chunk_ids: list[str] = []
        for hit in hits:
            try:
                record = record_from_qdrant_payload(hit.payload)
            except ValueError:
                missing_chunk_ids.append(hit.chunk_id)
                continue
            records[record.chunk_id] = record

        if missing_chunk_ids:
            records.update(self._fetch_records_from_qdrant_ids(missing_chunk_ids))
        return records

    def _fetch_records(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedRecord]:
        ordered_ids = dedupe_preserve_order(chunk_ids)
        if not ordered_ids:
            return {}

        if self._uses_qdrant_payload_records():
            return self._fetch_records_from_qdrant_ids(ordered_ids)

        if self._sqlite is None:
            raise RuntimeError("SQLite record store is not open.")

        placeholders = ", ".join("?" for _ in ordered_ids)
        rows = self._sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE chunk_id IN ({placeholders})
            """,
            ordered_ids,
        ).fetchall()

        return self._records_from_rows(rows)

    def _build_reference_payload_filter(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
    ):
        models = self._qdrant_models
        must_conditions: list[object] = []
        must_not_conditions: list[object] = []

        def add_match_any(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            must_conditions.append(
                models.FieldCondition(
                    key=field_name,
                    match=models.MatchAny(any=list(ordered_values)),
                )
            )

        add_match_any("document_id", document_ids)
        add_match_any("article_number", article_numbers)
        add_match_any("clause_ref", clause_refs)
        add_match_any("point_ref", point_refs)

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            must_not_conditions.append(
                models.FieldCondition(
                    key="chunk_id",
                    match=models.MatchAny(any=list(excluded_ids)),
                )
            )

        if not must_conditions and not must_not_conditions:
            return None
        return models.Filter(
            must=must_conditions or None,
            must_not=must_not_conditions or None,
        )

    def _fetch_records_by_reference_from_qdrant(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        query_filter = self._build_reference_payload_filter(
            document_ids=document_ids,
            article_numbers=article_numbers,
            clause_refs=clause_refs,
            point_refs=point_refs,
            exclude_chunk_ids=exclude_chunk_ids,
        )
        if query_filter is None:
            return ()

        points, _ = self._qdrant.scroll(
            collection_name=self._collection_name,
            scroll_filter=query_filter,
            limit=max(1, int(limit)) * 4,
            with_payload=True,
            with_vectors=False,
        )
        records = tuple(self._records_from_qdrant_points(points).values())
        return tuple(sorted(records, key=record_reference_sort_key)[: max(1, int(limit))])

    def _fetch_records_by_reference(
        self,
        *,
        document_ids: Sequence[str] = (),
        article_numbers: Sequence[str] = (),
        clause_refs: Sequence[str] = (),
        point_refs: Sequence[str] = (),
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 12,
    ) -> tuple[RetrievedRecord, ...]:
        if self._uses_qdrant_payload_records():
            return self._fetch_records_by_reference_from_qdrant(
                document_ids=document_ids,
                article_numbers=article_numbers,
                clause_refs=clause_refs,
                point_refs=point_refs,
                exclude_chunk_ids=exclude_chunk_ids,
                limit=limit,
            )

        if self._sqlite is None:
            raise RuntimeError("SQLite record store is not open.")

        where_parts: list[str] = []
        params: list[object] = []

        def add_in_filter(field_name: str, values: Sequence[str]) -> None:
            ordered_values = dedupe_preserve_order(tuple(value for value in values if value))
            if not ordered_values:
                return
            placeholders = ", ".join("?" for _ in ordered_values)
            where_parts.append(f"{field_name} IN ({placeholders})")
            params.extend(ordered_values)

        add_in_filter("document_id", document_ids)
        add_in_filter("article_number", article_numbers)
        add_in_filter("clause_ref", clause_refs)
        add_in_filter("point_ref", point_refs)

        excluded_ids = dedupe_preserve_order(tuple(value for value in exclude_chunk_ids if value))
        if excluded_ids:
            placeholders = ", ".join("?" for _ in excluded_ids)
            where_parts.append(f"chunk_id NOT IN ({placeholders})")
            params.extend(excluded_ids)

        if not where_parts:
            return ()

        params.append(max(1, int(limit)))
        rows = self._sqlite.execute(
            f"""
            SELECT chunk_id, parent_chunk_id, citation_text, text, dense_text, sparse_text, payload_json
            FROM records
            WHERE {" AND ".join(where_parts)}
            ORDER BY
                CASE level
                    WHEN 'article' THEN 0
                    WHEN 'clause' THEN 1
                    WHEN 'point' THEN 2
                    ELSE 3
                END,
                CAST(NULLIF(clause_ref, '') AS INTEGER),
                clause_ref,
                point_ref,
                chunk_id
            LIMIT ?
            """,
            params,
        ).fetchall()
        return tuple(self._records_from_rows(rows).values())

    def _fetch_article_siblings(
        self,
        *,
        document_id: str,
        article_number: str,
        exclude_chunk_ids: Sequence[str] = (),
        limit: int = 6,
    ) -> tuple[RetrievedRecord, ...]:
        return self._fetch_records_by_reference(
            document_ids=(document_id,),
            article_numbers=(article_number,),
            exclude_chunk_ids=exclude_chunk_ids,
            limit=limit,
        )

    @staticmethod
    def _record_to_search_hit(record: RetrievedRecord, score: float) -> SearchHit:
        return SearchHit(
            chunk_id=record.chunk_id,
            qdrant_point_id=str(record.payload.get("qdrant_point_id") or record.chunk_id),
            score=score,
            citation_text=record.citation_text,
            payload=record.payload,
        )

    def _append_reference_fallback_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        *,
        limit: int,
    ) -> tuple[SearchHit, ...]:
        if not intent.explicit_legal_reference_filters:
            return tuple(hits)

        existing_chunk_ids = {hit.chunk_id for hit in hits}
        existing_records = self._fetch_records_from_hits(hits)
        existing_articles = {
            str(record.payload.get("article_number") or "")
            for record in existing_records.values()
        }
        missing_articles = tuple(
            article for article in intent.article_numbers if article not in existing_articles
        )
        if intent.article_numbers and not missing_articles and not intent.clause_refs and not intent.point_refs:
            return tuple(hits)

        fallback_records = self._fetch_records_by_reference(
            document_ids=intent.document_filters,
            article_numbers=missing_articles or intent.article_numbers,
            clause_refs=intent.clause_refs,
            point_refs=intent.point_refs,
            exclude_chunk_ids=tuple(existing_chunk_ids),
            limit=limit,
        )
        if not fallback_records:
            return tuple(hits)

        fallback_score = min((hit.score for hit in hits), default=0.0) - 1e-3
        fallback_hits = tuple(
            self._record_to_search_hit(record, fallback_score - (rank * 1e-4))
            for rank, record in enumerate(fallback_records)
        )
        return (*hits, *fallback_hits)

    def _score_hit_relevance(
        self,
        hit: SearchHit,
        record: RetrievedRecord,
        intent: QueryIntent,
    ) -> float:
        boost = 0.0
        prioritized_issue_filters = prioritize_issue_filters(intent.issue_filters)
        article_number = str(record.payload.get("article_number") or "").lower()
        clause_ref = str(record.payload.get("clause_ref") or "").lower()
        document_id = str(record.payload.get("document_id") or "")
        level = str(record.payload.get("level") or "")
        actor_values = {str(value) for value in (record.payload.get("actor") or [])}
        topic_values = {str(value) for value in (record.payload.get("topic") or [])}
        issue_values = {str(value) for value in (record.payload.get("issue_type") or [])}
        section_heading = normalize_for_matching(str(record.payload.get("section_heading") or ""))
        heading_text = normalize_for_matching(
            " ".join(
                part
                for part in [
                    str(record.payload.get("article_title") or ""),
                    str(record.payload.get("heading") or ""),
                    record.citation_text,
                ]
                if part
            )
        )
        normalized_text = normalize_for_matching(f" {record.citation_text} {record.text} ")
        is_calculation_query = (
            contains_normalized_phrase(intent.normalized_query, CALCULATION_QUERY_HINTS)
            or (
                "tro_cap_thoi_viec" in intent.issue_filters
                and (
                    contains_normalized_phrase(intent.normalized_query, BENEFIT_COMPUTATION_QUERY_HINTS)
                    or YEAR_COUNT_RE.search(intent.normalized_query) is not None
                )
            )
        )
        wants_implementation_detail = contains_normalized_phrase(
            intent.normalized_query,
            IMPLEMENTATION_DETAIL_HINTS,
        )
        is_termination_query = (
            "cham_dut_hop_dong_lao_dong" in intent.topic_filters
            or "tro_cap_thoi_viec" in intent.issue_filters
            or "tro_cap_mat_viec" in intent.issue_filters
            or "nghia_vu_khi_cham_dut" in intent.issue_filters
            or contains_normalized_phrase(intent.normalized_query, TERMINATION_QUERY_HINTS)
        )
        is_termination_benefit_query = (
            is_termination_query
            and (
                "tro_cap" in intent.topic_filters
                or "tro_cap_thoi_viec" in intent.issue_filters
                or "tro_cap_mat_viec" in intent.issue_filters
            )
        )
        query_has_maternity_hint = contains_normalized_phrase(intent.normalized_query, MATERNITY_CONTEXT_HINTS)
        query_has_retirement_hint = contains_normalized_phrase(intent.normalized_query, RETIREMENT_CONTEXT_HINTS)
        prefers_primary_law = (
            is_termination_query
            and not wants_implementation_detail
            and "nghi dinh" not in intent.normalized_query
        )

        if intent.all_article_numbers and article_number in intent.all_article_numbers:
            boost += 0.2
        if intent.clause_refs and clause_ref in intent.clause_refs:
            boost += 0.25
        if prioritized_issue_filters and issue_values.intersection(prioritized_issue_filters):
            boost += 0.15
        if intent.topic_filters and topic_values.intersection(intent.topic_filters):
            boost += 0.05
        specific_actor_filters = filter_specific_actor_labels(intent.actor_filters)
        if specific_actor_filters and actor_values.intersection(specific_actor_filters):
            boost += 0.04
        if "time_limit" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("ngay", "thang", "nam", "thoi han", "khong qua", "it nhat"),
        ):
            boost += 0.08
        if "money_percentage" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("%", "phan tram", "tien luong", "it nhat", "bang", "muc luong"),
        ):
            boost += 0.08
        if "procedure" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("thong bao", "tham khao y kien", "bien ban", "thoi han", "to chuc dai dien"),
        ):
            boost += 0.08
        if "classification" in intent.query_types and contains_normalized_phrase(
            normalized_text,
            ("hinh thuc", "don phuong cham dut", "cac truong hop", "ap dung"),
        ):
            boost += 0.08

        if is_termination_query:
            if contains_normalized_phrase(section_heading, TERMINATION_SECTION_HINTS):
                boost += 0.25
            if "cham_dut_hop_dong_lao_dong" in topic_values:
                boost += 0.15
            if "nghia_vu_khi_cham_dut" in issue_values:
                boost += 0.1

        if is_termination_benefit_query:
            if issue_values.intersection({"tro_cap_thoi_viec", "tro_cap_mat_viec"}):
                boost += 0.35
            if contains_normalized_phrase(normalized_text, TERMINATION_BENEFIT_CONTEXT_HINTS):
                boost += 0.15
            if "bao hiem that nghiep" in intent.normalized_query and "bao hiem that nghiep" in normalized_text:
                boost += 0.25
            if contains_normalized_phrase(heading_text, MATERNITY_CONTEXT_HINTS) and not query_has_maternity_hint:
                boost -= 0.7
            if contains_normalized_phrase(heading_text, RETIREMENT_CONTEXT_HINTS) and not query_has_retirement_hint:
                boost -= 0.45
            if "tro_cap_thoi_viec" in prioritized_issue_filters:
                if "tro cap thoi viec" in heading_text:
                    boost += 0.4
                if "tro cap mat viec" in heading_text and "tro cap thoi viec" not in heading_text:
                    boost -= 0.2
                if "nghia vu" in heading_text or "trach nhiem" in heading_text:
                    boost -= 0.15
            if "tro_cap_mat_viec" in prioritized_issue_filters:
                if "tro cap mat viec" in heading_text:
                    boost += 0.4
                if "tro cap thoi viec" in heading_text and "tro cap mat viec" not in heading_text:
                    boost -= 0.2

        if not wants_implementation_detail and not intent.point_refs and not intent.clause_refs:
            if level == "clause":
                boost += 0.12
            elif level == "point":
                boost -= 0.35

        if prefers_primary_law:
            if document_id == "45-2019-qh14":
                boost += 0.22
            elif document_id == "nghi-dinh-145-2020-nd-cp":
                boost -= 0.08
                if level == "point":
                    boost -= 0.18

        if is_calculation_query:
            if contains_normalized_phrase(normalized_text, CALCULATION_CONTEXT_HINTS):
                boost += 0.25
            if (
                contains_normalized_phrase(normalized_text, DELEGATION_CONTEXT_HINTS)
                and not wants_implementation_detail
            ):
                boost -= 0.9

        return hit.score + boost

    def _rerank_hits(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        scored_hits: list[tuple[float, SearchHit]] = []
        for hit in hits:
            record = direct_records.get(hit.chunk_id)
            adjusted_score = self._score_hit_relevance(hit, record, intent) if record is not None else hit.score
            scored_hits.append((adjusted_score, hit))

        ordered = sorted(
            scored_hits,
            key=lambda item: (-item[0], -item[1].score, item[1].citation_text),
        )
        return tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=adjusted_score,
                citation_text=hit.citation_text,
                payload=hit.payload,
            )
            for adjusted_score, hit in ordered
        )

    def _predict_reranker_scores(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> dict[str, float]:
        reranker = self._get_reranker()
        if reranker is None:
            return {}

        hit_records: list[tuple[SearchHit, RetrievedRecord]] = []
        for hit in hits[: self._reranker_top_n]:
            record = direct_records.get(hit.chunk_id)
            if record is None:
                continue
            hit_records.append((hit, record))

        if not hit_records:
            return {}

        pairs = [
            (
                query,
                record.dense_text.strip() or f"{record.citation_text}\n{record.text}".strip(),
            )
            for _, record in hit_records
        ]
        raw_scores = reranker.predict(pairs, show_progress_bar=False)
        if hasattr(raw_scores, "tolist"):
            raw_scores = raw_scores.tolist()

        def coerce_score(value: object) -> float:
            if isinstance(value, (list, tuple)):
                return float(value[0])
            if hasattr(value, "item"):
                return float(value.item())
            return float(value)

        return {
            hit.chunk_id: coerce_score(score)
            for (hit, _), score in zip(hit_records, raw_scores)
        }

    def _semantic_rerank_hits(
        self,
        query: str,
        hits: Sequence[SearchHit],
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[SearchHit, ...]:
        if not self.reranker_enabled:
            return tuple(hits)

        candidate_hits = tuple(hits[: self._reranker_top_n])
        if not candidate_hits:
            return tuple(hits)

        semantic_scores = self._predict_reranker_scores(query, candidate_hits, direct_records)
        if not semantic_scores:
            return tuple(hits)

        heuristic_rank = {hit.chunk_id: rank for rank, hit in enumerate(candidate_hits, start=1)}
        semantic_rank = {
            hit.chunk_id: rank
            for rank, hit in enumerate(
                sorted(
                    candidate_hits,
                    key=lambda current_hit: (
                        -semantic_scores.get(current_hit.chunk_id, float("-inf")),
                        -current_hit.score,
                        current_hit.citation_text,
                    ),
                ),
                start=1,
            )
        }

        def fused_rrf_score(hit: SearchHit) -> float:
            return (
                1.0 / (RRF_K + heuristic_rank[hit.chunk_id])
                + 1.0 / (RRF_K + semantic_rank[hit.chunk_id])
            )

        fused_candidates = sorted(
            candidate_hits,
            key=lambda hit: (
                -fused_rrf_score(hit),
                -semantic_scores.get(hit.chunk_id, float("-inf")),
                -hit.score,
                hit.citation_text,
            ),
        )

        max_existing_score = max((hit.score for hit in hits), default=0.0)
        score_step = 1e-4
        reranked_candidates = tuple(
            SearchHit(
                chunk_id=hit.chunk_id,
                qdrant_point_id=hit.qdrant_point_id,
                score=max_existing_score + 1.0 - (rank * score_step),
                citation_text=hit.citation_text,
                payload=hit.payload,
            )
            for rank, hit in enumerate(fused_candidates)
        )
        remainder = tuple(hits[len(candidate_hits) :])
        return reranked_candidates + remainder

    def _query_needs_article_siblings(self, intent: QueryIntent) -> bool:
        if intent.clause_refs or intent.point_refs:
            return True
        if set(intent.query_types).intersection({"procedure", "remedy", "definition", "classification", "money_percentage"}):
            return True
        return bool(
            set(intent.issue_filters).intersection(
                {
                    "boi_thuong",
                    "tro_cap_thoi_viec",
                    "tro_cap_mat_viec",
                    "nghia_vu_khi_cham_dut",
                    "trai_phap_luat",
                }
            )
        )

    def _add_article_sibling_contexts(
        self,
        contexts: Sequence[RetrievalContext],
        *,
        intent: QueryIntent,
        direct_records: dict[str, RetrievedRecord],
    ) -> tuple[RetrievalContext, ...]:
        if not self._query_needs_article_siblings(intent):
            return tuple(contexts)

        seen_chunk_ids = {context.chunk_id for context in contexts}
        seen_chunk_ids.update(
            chunk_id
            for context in contexts
            for chunk_id in context.matched_chunk_ids
        )
        context_article_levels = {
            context_article_key(context): str(context.payload.get("level") or "")
            for context in contexts
            if context_article_key(context) is not None
        }
        records_by_article: dict[tuple[str, str], RetrievedRecord] = {}
        for record in direct_records.values():
            document_id = str(record.payload.get("document_id") or "")
            article_number = str(record.payload.get("article_number") or "")
            if not document_id or not article_number:
                continue
            key = (document_id, article_number)
            if context_article_levels.get(key) == "article":
                continue
            records_by_article.setdefault(key, record)

        if not records_by_article:
            return tuple(contexts)

        expanded_contexts: list[RetrievalContext] = []
        added_sibling_ids: set[str] = set()
        for context in contexts:
            expanded_contexts.append(context)
            key = context_article_key(context)
            if key not in records_by_article:
                continue
            document_id, article_number = key
            siblings = self._fetch_article_siblings(
                document_id=document_id,
                article_number=article_number,
                exclude_chunk_ids=tuple(seen_chunk_ids | added_sibling_ids),
                limit=3,
            )
            for rank, sibling in enumerate(siblings, start=1):
                added_sibling_ids.add(sibling.chunk_id)
                expanded_contexts.append(
                    RetrievalContext(
                        chunk_id=sibling.chunk_id,
                        citation_text=sibling.citation_text,
                        text=sibling.text,
                        payload=sibling.payload,
                        score=max(context.score - (rank * 1e-3), 0.0),
                        matched_chunk_ids=(sibling.chunk_id,),
                        matched_citations=(sibling.citation_text,),
                    )
                )

        return tuple(expanded_contexts)

    def _assemble_contexts(
        self,
        hits: Sequence[SearchHit],
        intent: QueryIntent | None = None,
    ) -> tuple[RetrievalContext, ...]:
        direct_records = self._fetch_records_from_hits(hits)
        parent_ids = [
            record.parent_chunk_id
            for record in direct_records.values()
            if record.parent_chunk_id and record.parent_chunk_id not in direct_records
        ]
        parent_records = self._fetch_records(parent_ids)
        record_map = {**direct_records, **parent_records}

        grouped: dict[str, dict[str, object]] = {}
        for rank, hit in enumerate(hits):
            matched_record = direct_records.get(hit.chunk_id)
            if matched_record is None:
                continue

            context_id = matched_record.parent_chunk_id or matched_record.chunk_id
            context_record = record_map.get(context_id, matched_record)
            group = grouped.setdefault(
                context_record.chunk_id,
                {
                    "record": context_record,
                    "score": hit.score,
                    "rank": rank,
                    "matched_chunk_ids": [],
                    "matched_citations": [],
                },
            )
            group["score"] = max(float(group["score"]), hit.score)
            group["rank"] = min(int(group["rank"]), rank)
            group["matched_chunk_ids"].append(hit.chunk_id)
            group["matched_citations"].append(hit.citation_text)

        ordered_groups = sorted(
            grouped.values(),
            key=lambda item: (-float(item["score"]), int(item["rank"])),
        )

        contexts: list[RetrievalContext] = []
        for item in ordered_groups:
            record = item["record"]
            assert isinstance(record, RetrievedRecord)
            matched_chunk_ids = dedupe_preserve_order(item["matched_chunk_ids"])
            matched_citations = dedupe_preserve_order(item["matched_citations"])
            contexts.append(
                RetrievalContext(
                    chunk_id=record.chunk_id,
                    citation_text=record.citation_text,
                    text=record.text,
                    payload=record.payload,
                    score=float(item["score"]),
                    matched_chunk_ids=matched_chunk_ids,
                    matched_citations=matched_citations,
                )
            )
        if intent is None:
            return tuple(contexts)
        return self._add_article_sibling_contexts(
            contexts,
            intent=intent,
            direct_records=direct_records,
        )

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        prefetch_limit: int = 24,
    ) -> RetrievalResult:
        intent = route_query(query)
        query_filter = self._build_query_filter(intent)
        reference_boost_filter = self._build_reference_boost_filter(intent)
        issue_focus_filter = self._build_issue_focus_filter(intent)
        query_variants = build_query_variants(intent)
        _, sparse_query = self._encode_sparse_query(intent)
        models = self._qdrant_models

        prefetches = []
        for variant in query_variants:
            dense_query = self._encode_dense_query(variant)
            _, variant_sparse_query = self._encode_sparse_query(intent, query_text=variant)
            prefetches.extend(
                [
                    models.Prefetch(
                        query=dense_query,
                        using=self._dense_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                    models.Prefetch(
                        query=variant_sparse_query,
                        using=self._sparse_vector_name,
                        filter=query_filter,
                        limit=prefetch_limit,
                    ),
                ]
            )

        if reference_boost_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self._sparse_vector_name,
                    filter=reference_boost_filter,
                    limit=max(8, prefetch_limit // 2),
                )
            )
        if issue_focus_filter is not None:
            prefetches.append(
                models.Prefetch(
                    query=sparse_query,
                    using=self._sparse_vector_name,
                    filter=issue_focus_filter,
                    limit=max(12, prefetch_limit),
                )
            )

        candidate_limit = max(top_k * 6, prefetch_limit * max(4, len(query_variants) * 2), 64)

        response = self._qdrant.query_points(
            collection_name=self._collection_name,
            prefetch=prefetches,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=candidate_limit,
            with_payload=True,
        )

        hits = tuple(
            SearchHit(
                chunk_id=str(point.payload["chunk_id"]),
                qdrant_point_id=str(
                    point.payload.get("qdrant_point_id")
                    or make_qdrant_point_id(str(point.payload["chunk_id"]))
                ),
                score=float(point.score),
                citation_text=str(point.payload.get("citation_text") or ""),
                payload=dict(point.payload),
            )
            for point in response.points
        )
        hits = self._append_reference_fallback_hits(hits, intent, limit=max(4, top_k))
        direct_records = self._fetch_records_from_hits(hits)
        hits = self._rerank_hits(hits, intent, direct_records)
        hits = self._semantic_rerank_hits(query, hits, direct_records)[:top_k]
        contexts = self._assemble_contexts(hits, intent=intent)
        return RetrievalResult(
            query=query,
            intent=intent,
            hits=hits,
            contexts=contexts,
        )


__all__ = [
    "HybridRetriever",
    "QueryIntent",
    "RetrievalContext",
    "RetrievalResult",
    "LEGAL_ISSUE_ARTICLE_MAP",
    "TERMINATION_ARTICLE_MAP",
    "SearchHit",
    "RetrievedRecord",
    "build_context_block",
    "build_query_variants",
    "dedupe_preserve_order",
    "DEFAULT_MAX_CONTEXT_CHARS",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "DEFAULT_RERANKER_TOP_N",
    "RECORD_SOURCE_QDRANT_PAYLOAD",
    "RECORD_SOURCE_SQLITE",
    "diversify_contexts_by_article",
    "estimate_token_count",
    "format_context_for_prompt",
    "format_intent_summary",
    "load_manifest",
    "parse_reference_values",
    "record_from_qdrant_payload",
    "resolve_record_source",
    "route_query",
    "select_contexts_for_prompt",
]
