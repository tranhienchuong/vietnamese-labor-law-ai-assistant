from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Sequence

from .corpus_pipeline import normalize_for_matching


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
    ),
}

MAX_ENUMERATION_CONTEXT_RECORDS = 16
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
ENUMERATION_QUERY_HINTS = (
    "cac truong hop",
    "nhung truong hop",
    "truong hop nao",
    "khi nao duoc",
    "duoc trong truong hop nao",
    "gom nhung gi",
    "bao gom nhung gi",
    "liet ke",
)
ENUMERATION_PARENT_CONTEXT_HINTS = (
    "sau day:",
    "sau day",
    "nhu sau:",
    "bao gom:",
)
NO_NOTICE_QUERY_HINTS = (
    "khong can bao truoc",
    "khong can phai bao truoc",
    "khong phai bao truoc",
    "khong bao truoc",
    "nghi ngay",
    "nghi viec ngay",
    "nghi luon",
)


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
    "enumeration": (
        "cac truong hop",
        "nhung truong hop",
        "truong hop nao",
        "gom nhung gi",
        "bao gom nhung gi",
        "liet ke",
    ),
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


def query_asks_for_enumeration(intent: QueryIntent) -> bool:
    if "enumeration" in intent.query_types:
        return True
    if contains_normalized_phrase(intent.normalized_query, ENUMERATION_QUERY_HINTS):
        return True
    return "can_cu_cham_dut" in intent.issue_filters and contains_normalized_phrase(
        intent.normalized_query,
        ("khi nao", "truong hop", "can cu", "ly do"),
    )


def query_asks_without_notice(intent: QueryIntent) -> bool:
    return contains_normalized_phrase(intent.normalized_query, NO_NOTICE_QUERY_HINTS)


def infer_employee_notice_period_reference(
    intent: QueryIntent,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if "35" not in intent.all_article_numbers:
        return (), ()

    if (
        "thoi_han_bao_truoc" not in intent.issue_filters
        and "bao_truoc" not in intent.topic_filters
        and "time_limit" not in intent.query_types
    ):
        return (), ()

    query = intent.normalized_query
    if query_asks_without_notice(intent):
        return ("2",), ()
    if "khong xac dinh thoi han" in query:
        return ("1",), ("a",)
    if "duoi 12" in query:
        return ("1",), ("c",)
    if "12" in query and "36" in query and "xac dinh thoi han" in query:
        return ("1",), ("b",)
    if contains_normalized_phrase(query, ("dac thu", "nganh nghe", "cong viec dac thu")):
        return ("1",), ("d",)

    return ("1",), ("a", "b", "c", "d")


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


def route_query_heuristic(query: str) -> QueryIntent:
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


route_query = route_query_heuristic


__all__ = [
    "ARTICLE_REF_RE",
    "BENEFIT_COMPUTATION_QUERY_HINTS",
    "CALCULATION_CONTEXT_HINTS",
    "CALCULATION_QUERY_HINTS",
    "DELEGATION_CONTEXT_HINTS",
    "ENUMERATION_PARENT_CONTEXT_HINTS",
    "IMPLEMENTATION_DETAIL_HINTS",
    "LEGAL_ISSUE_ARTICLE_MAP",
    "LEGAL_ISSUE_QUERY_HINTS",
    "LEGAL_TOPIC_ARTICLE_MAP",
    "MATERNITY_CONTEXT_HINTS",
    "MAX_ENUMERATION_CONTEXT_RECORDS",
    "NO_NOTICE_QUERY_HINTS",
    "QueryIntent",
    "RETIREMENT_CONTEXT_HINTS",
    "RuleBasedQueryExpansion",
    "TERMINATION_ARTICLE_MAP",
    "TERMINATION_BENEFIT_CONTEXT_HINTS",
    "TERMINATION_QUERY_HINTS",
    "TERMINATION_SECTION_HINTS",
    "YEAR_COUNT_RE",
    "build_query_variants",
    "collect_keyword_matches",
    "collect_rule_based_query_expansions",
    "contains_normalized_phrase",
    "dedupe_preserve_order",
    "filter_specific_actor_labels",
    "format_intent_summary",
    "infer_employee_notice_period_reference",
    "parse_reference_values",
    "prioritize_issue_filters",
    "query_asks_for_enumeration",
    "query_asks_without_notice",
    "route_query",
    "route_query_heuristic",
]
