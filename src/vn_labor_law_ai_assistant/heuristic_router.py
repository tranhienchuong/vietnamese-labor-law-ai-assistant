from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Sequence

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
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "chua thanh nien",
        "chua du 15 tuoi",
        "14 tuoi",
        "tre 14 tuoi",
        "be 14 tuoi",
        "hoc sinh lam them",
        "13 tuoi",
    ),
    "nguoi_lao_dong_nuoc_ngoai": (
        "lao dong nuoc ngoai",
        "nguoi nuoc ngoai",
        "nhan vien nuoc ngoai",
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
    "tuyen_dung_lao_dong": (
        "tuyen dung",
        "ung vien",
        "khong tuyen",
        "khong nhan ung vien",
        "tu choi ung vien",
        "loai ung vien",
        "loai ho so",
        "sang loc ung vien",
    ),
    "thoi_gio_lam_viec": (
        "thoi gio lam viec",
        "lam them gio",
        "ca dem",
        "ca khuya",
        "lam ban dem",
        "lam khuya",
        "nua dem",
        "22h",
        "23h",
        "5h sang",
        "2h sang",
    ),
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "chua du 15 tuoi",
        "14 tuoi",
        "tre 14 tuoi",
        "be 14 tuoi",
        "hoc sinh lam them",
        "13 tuoi",
    ),
    "cho_thue_lai_lao_dong": (
        "cho thue lai lao dong",
        "lao dong thue lai",
        "thue lai lao dong",
    ),
    "tranh_chap_lao_dong": (
        "tranh chap lao dong",
        "tranh chap",
        "khoi kien",
        "toa an",
        "hoa giai vien lao dong",
    ),
    "binh_dang_phan_biet_doi_xu": (
        "phan biet doi xu",
        "khong tuyen phu nu",
        "khong nhan ung vien",
        "tu choi ung vien",
        "hiv",
        "quay roi tinh duc",
    ),
}

ISSUE_KEYWORDS = {
    "can_cu_cham_dut": (
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
    "giai_thich_tu_ngu": (
        "dinh nghia",
        "duoc hieu la",
        "la ai",
        "la gi",
        "the nao la",
    ),
    "quyen_nghia_vu_nguoi_lao_dong": (
        "quyen va nghia vu cua nguoi lao dong",
        "quyen nghia vu cua nguoi lao dong",
    ),
    "quyen_nghia_vu_nguoi_su_dung_lao_dong": (
        "quyen va nghia vu cua nguoi su dung lao dong",
        "quyen nghia vu cua nguoi su dung lao dong",
    ),
    "hanh_vi_bi_cam": (
        "hanh vi bi nghiem cam",
        "hanh vi bi cam",
        "cam trong linh vuc lao dong",
    ),
    "hanh_vi_cam_khi_giao_ket": (
        "hanh vi nguoi su dung lao dong khong duoc lam",
        "khong duoc lam khi giao ket",
    ),
    "giu_giay_to_goc": (
        "giu cccd",
        "giu can cuoc",
        "giu can cuoc cong dan",
        "giu cmnd",
        "giu ho chieu",
        "giu passport",
        "giu giay to goc",
        "giu giay to ca nhan",
        "giu ban chinh",
        "giu bang dai hoc",
        "giu bang cap goc",
        "giu van bang goc",
        "giu chung chi goc",
        "giu van bang",
        "giu chung chi",
        "giay to tuy than",
        "giay to tuy than ban goc",
    ),
    "dat_coc_bao_dam": (
        "dat coc",
        "nop tien coc",
        "tien the chan",
        "tien giu chan",
        "tien bao lanh",
        "ky quy",
        "bao dam bang tien",
        "bao dam bang tai san",
        "giu tien de bao dam",
    ),
    "phan_biet_doi_xu": (
        "phan biet doi xu",
        "khong tuyen phu nu",
        "khong nhan ung vien",
        "tu choi ung vien",
        "phu nu co con nho",
        "co con nho",
        "nhiem hiv",
        "hiv",
        "loai ung vien",
    ),
    "quay_roi_tinh_duc": (
        "quay roi tinh duc",
        "ga gam",
        "goi y tinh duc",
        "nhan tin ga gam",
    ),
    "tuyen_dung_lao_dong": (
        "tuyen dung",
        "ung vien",
        "khong tuyen",
        "loai ung vien",
    ),
    "thoi_gio_lam_viec": (
        "thoi gio lam viec",
        "ca dem",
        "lam ban dem",
        "lam them gio",
    ),
    "lam_ban_dem": (
        "ca dem",
        "ca khuya",
        "lam ban dem",
        "lam khuya",
        "lam den nua dem",
        "qua dem",
        "22h",
        "23h",
        "nua dem",
        "5h sang",
        "2h sang",
    ),
    "lam_them_gio": (
        "lam them gio",
        "tang ca",
    ),
    "lao_dong_chua_thanh_nien": (
        "lao dong chua thanh nien",
        "chua du 15 tuoi",
        "14 tuoi",
        "tre 14 tuoi",
        "be 14 tuoi",
        "hoc sinh lam them",
        "13 tuoi",
    ),
    "cho_thue_lai_lao_dong": (
        "cho thue lai lao dong",
        "lao dong thue lai",
        "thue lai lao dong",
    ),
    "tranh_chap_lao_dong": (
        "tranh chap lao dong",
        "tranh chap",
        "khoi kien",
        "toa an",
        "hoa giai vien lao dong",
    ),
    "du_lieu_ca_nhan": (
        "du lieu ca nhan",
        "thong tin ca nhan",
        "ho so ca nhan",
    ),
    "thong_tin_suc_khoe": (
        "thong tin suc khoe",
        "ho so suc khoe",
        "benh an",
        "tinh trang hiv",
    ),
    "ep_nghi_viec": (
        "ep nghi",
        "ep ky don",
        "bat toi ky don",
        "bat ky don nghi viec",
        "ky don tu nguyen nghi viec",
        "don tu nguyen nghi viec",
        "ep viet don nghi",
        "bat viet don nghi",
        "doa cho nghi",
        "gay ap luc nghi viec",
    ),
    "dieu_khoan_bat_cong": (
        "dieu khoan bat cong",
        "dieu khoan bat loi",
        "thoa thuan bat cong",
        "quyen loi thap hon luat",
    ),
    "han_che_viec_lam_sau_nghi": (
        "khong duoc lam cung nganh",
        "cam lam cung nganh",
        "khong duoc lam cho doi thu",
        "sau khi nghi viec",
        "sau khi nghi",
    ),
    "bao_mat_bi_mat_kinh_doanh": (
        "bao mat",
        "bi mat kinh doanh",
        "bi mat cong nghe",
        "thoa thuan bao mat",
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
    context_phrases: tuple[str, ...] = ()
    confidence: Literal["high", "medium", "low"] = "medium"


@dataclass(frozen=True)
class RuleBasedRoutingResult:
    inferred_articles: tuple[str, ...]
    force_reference_articles: tuple[str, ...]
    topics: tuple[str, ...]
    issues: tuple[str, ...]
    expansions: tuple[str, ...]


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

LEGAL_HIGH_PRECISION_QUERY_RULES = (
    RuleBasedQueryExpansion(
        phrases=(
            "nguoi lao dong duoc dinh nghia",
            "nguoi lao dong la ai",
            "nguoi su dung lao dong la ai",
            "quan he lao dong bao gom",
        ),
        articles=("3",),
        topics=("general_provisions",),
        issues=("giai_thich_tu_ngu",),
        expansions=("Dieu 3 giai thich tu ngu nguoi lao dong nguoi su dung lao dong quan he lao dong",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("hop dong lao dong la gi", "the nao la hop dong lao dong", "dinh nghia hop dong lao dong"),
        articles=("13",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong", "giai_thich_tu_ngu"),
        expansions=("Dieu 13 hop dong lao dong la su thoa thuan ve viec lam co tra cong tien luong",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("khong xac dinh thoi han", "xac dinh thoi han", "loai hop dong lao dong", "co may loai hop dong"),
        articles=("20",),
        topics=("hop_dong_lao_dong",),
        issues=("loai_hop_dong",),
        expansions=("Dieu 20 loai hop dong lao dong khong xac dinh thoi han xac dinh thoi han",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("hinh thuc hop dong lao dong", "hop dong lao dong phai duoc lap", "lap thanh may ban", "may ban hop dong"),
        articles=("14",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=("Dieu 14 hinh thuc hop dong lao dong bang van ban lap thanh 02 ban",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("bang loi noi", "hop dong lao dong bang loi noi", "giao ket hop dong lao dong bang loi noi", "giao ket hop dong bang loi noi"),
        articles=("14", "18", "145", "162"),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=("Dieu 14 hop dong lao dong bang loi noi doi voi hop dong co thoi han duoi 01 thang",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("noi dung chu yeu cua hop dong lao dong", "noi dung hop dong lao dong gom", "noi dung bat buoc trong hop dong lao dong", "noi dung bat buoc cua hop dong lao dong"),
        articles=("21",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=("Dieu 21 noi dung hop dong lao dong",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("phu luc hop dong lao dong", "moi quan he giua hop dong lao dong va phu luc"),
        articles=("22",),
        topics=("hop_dong_lao_dong",),
        issues=("sua_doi_bo_sung_hop_dong",),
        expansions=("Dieu 22 phu luc hop dong lao dong",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("hop dong thu viec", "noi dung thu viec", "thu viec ghi trong hop dong lao dong"),
        articles=("24", "27"),
        topics=("thu_viec", "hop_dong_lao_dong"),
        issues=("thu_viec",),
        expansions=("Dieu 24 thu viec Dieu 27 ket thuc thoi gian thu viec",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "giu cccd",
            "giu can cuoc",
            "giu can cuoc cong dan",
            "giu cmnd",
            "giu ho chieu",
            "giu passport",
            "giu giay to goc",
            "giu giay to ca nhan",
            "giu giay to tuy than",
            "giu ban chinh",
            "giu bang dai hoc",
            "giu bang cap goc",
            "giu van bang goc",
            "giu chung chi goc",
            "giu van bang",
            "giu chung chi",
            "giay to tuy than ban goc",
        ),
        articles=("17",),
        topics=("hop_dong_lao_dong",),
        issues=("hanh_vi_cam_khi_giao_ket", "giu_giay_to_goc"),
        expansions=(
            "Dieu 17 hanh vi nguoi su dung lao dong khong duoc lam",
            "giu ban chinh giay to tuy than van bang chung chi cua nguoi lao dong",
        ),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "dat coc",
            "nop tien coc",
            "tien the chan",
            "tien giu chan",
            "tien bao lanh",
            "ky quy",
            "bao dam bang tien",
            "bao dam bang tai san",
            "giu tien de bao dam",
        ),
        articles=("17",),
        topics=("hop_dong_lao_dong",),
        issues=("hanh_vi_cam_khi_giao_ket", "dat_coc_bao_dam"),
        expansions=("Dieu 17 yeu cau nguoi lao dong thuc hien bien phap bao dam bang tien hoac tai san",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("hiv", "nhiem hiv", "tinh trang hiv"),
        articles=("3", "8", "11", "16"),
        topics=("tuyen_dung_lao_dong", "binh_dang_phan_biet_doi_xu"),
        issues=("phan_biet_doi_xu", "tuyen_dung_lao_dong"),
        expansions=(
            "phan biet doi xu trong lao dong theo tinh trang HIV",
            "tuyen dung lao dong khong phan biet doi xu",
        ),
        context_phrases=(
            "tuyen dung",
            "ung vien",
            "khong tuyen",
            "khong nhan ung vien",
            "tu choi ung vien",
            "loai ung vien",
            "loai ho so",
            "sang loc ung vien",
        ),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "khong tuyen phu nu",
            "khong nhan ung vien nu",
            "tu choi ung vien nu",
            "loai ung vien vi co con nho",
            "loai ho so vi co con nho",
            "khong tuyen vi co con nho",
        ),
        articles=("3", "8", "11", "135"),
        topics=("tuyen_dung_lao_dong", "binh_dang_phan_biet_doi_xu", "bao_ve_thai_san"),
        issues=("phan_biet_doi_xu", "tuyen_dung_lao_dong"),
        expansions=("phan biet doi xu trong lao dong", "tuyen dung lao dong khong phan biet doi xu"),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("14 tuoi", "tre 14 tuoi", "be 14 tuoi", "13 tuoi", "chua du 15 tuoi", "hoc sinh lam them"),
        articles=("143", "145", "146", "106"),
        topics=("lao_dong_chua_thanh_nien", "thoi_gio_lam_viec"),
        issues=("lao_dong_chua_thanh_nien", "lam_ban_dem"),
        expansions=(
            "nguoi chua du 15 tuoi khong duoc lam them gio lam viec vao ban dem",
            "su dung nguoi chua du 15 tuoi",
        ),
        context_phrases=("ca dem", "ca khuya", "lam ban dem", "lam khuya", "lam den nua dem", "nua dem", "qua dem", "22h", "23h", "2h sang", "5h sang"),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("lao dong thue lai", "cho thue lai lao dong", "thue lai lao dong"),
        articles=("52", "53"),
        topics=("cho_thue_lai_lao_dong",),
        issues=("cho_thue_lai_lao_dong",),
        expansions=("khong duoc su dung lao dong thue lai de thay the nguoi lao dong dang dinh cong",),
        context_phrases=("nguoi lao dong dang dinh cong", "dang dinh cong"),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("phat tien", "cat luong thay viec xu ly ky luat", "di muon bi phat tien", "di muon bi tru", "tru tien luong vi di muon", "tru luong thay ky luat"),
        articles=("127", "118", "124"),
        topics=("ky_luat_sa_thai",),
        issues=("xu_ly_ky_luat_lao_dong", "noi_quy_lao_dong"),
        expansions=("Dieu 127 nghiem cam phat tien cat luong thay viec xu ly ky luat lao dong",),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("quay roi tinh duc", "ga gam", "goi y tinh duc", "nhan tin ga gam", "nhan tin dung tuc"),
        articles=("3", "8", "35", "118"),
        topics=("binh_dang_phan_biet_doi_xu", "ky_luat_sa_thai"),
        issues=("quay_roi_tinh_duc", "hanh_vi_bi_cam", "noi_quy_lao_dong"),
        expansions=(
            "quay roi tinh duc tai noi lam viec",
            "nguoi lao dong co quyen don phuong cham dut khong can bao truoc khi bi quay roi tinh duc",
            "noi quy lao dong phai co phong chong quay roi tinh duc",
        ),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("ep ky don", "ep ky don nghi viec", "bat toi ky don", "bat ky don nghi viec", "ky don tu nguyen nghi viec", "don tu nguyen nghi viec", "ep viet don nghi", "ep viet don xin nghi", "buoc viet don nghi", "bat viet don nghi", "ep nghi", "doa cho nghi", "gay ap luc nghi viec"),
        articles=("7", "15", "34", "36", "39", "41"),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("ep_nghi_viec", "trai_phap_luat", "boi_thuong"),
        expansions=(
            "quan he lao dong tu nguyen thien chi binh dang hop tac",
            "giao ket hop dong lao dong tu nguyen binh dang",
            "don phuong cham dut hop dong lao dong trai phap luat",
        ),
        confidence="high",
    ),
    RuleBasedQueryExpansion(
        phrases=("khong duoc lam cung nganh", "cam lam cung nganh", "khong duoc lam cho doi thu", "cam lam cho doi thu", "han che viec lam sau khi nghi", "khong duoc lam cung nganh suot doi", "lam cung nganh vinh vien"),
        articles=("10", "21", "15"),
        topics=("hop_dong_lao_dong",),
        issues=("han_che_viec_lam_sau_nghi", "bao_mat_bi_mat_kinh_doanh", "dieu_khoan_bat_cong"),
        expansions=(
            "quyen lam viec tu do lua chon viec lam",
            "thoa thuan bao ve bi mat kinh doanh bi mat cong nghe",
            "giao ket hop dong lao dong khong duoc trai phap luat",
        ),
        confidence="high",
    ),
)

LEGAL_SOFT_HINT_QUERY_RULES = (
    RuleBasedQueryExpansion(
        phrases=(
            "nguoi lao dong duoc dinh nghia",
            "nguoi lao dong la ai",
            "nguoi su dung lao dong la ai",
            "quan he lao dong bao gom",
        ),
        articles=("3",),
        topics=("general_provisions",),
        issues=("giai_thich_tu_ngu",),
        expansions=(
            "Dieu 3 giai thich tu ngu nguoi lao dong nguoi su dung lao dong quan he lao dong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "hop dong lao dong la gi",
            "the nao la hop dong lao dong",
            "dinh nghia hop dong lao dong",
        ),
        articles=("13",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong", "giai_thich_tu_ngu"),
        expansions=(
            "Dieu 13 hop dong lao dong la su thoa thuan ve viec lam co tra cong tien luong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "khong xac dinh thoi han",
            "xac dinh thoi han",
            "loai hop dong lao dong",
            "co may loai hop dong",
        ),
        articles=("20",),
        topics=("hop_dong_lao_dong",),
        issues=("loai_hop_dong",),
        expansions=("Dieu 20 loai hop dong lao dong khong xac dinh thoi han xac dinh thoi han",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "lap thanh may ban",
            "may ban hop dong",
            "hinh thuc hop dong lao dong",
            "hop dong lao dong phai duoc lap",
        ),
        articles=("14",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=("Dieu 14 hinh thuc hop dong lao dong bang van ban lap thanh 02 ban",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "bang loi noi",
            "hop dong lao dong bang loi noi",
            "giao ket hop dong lao dong bang loi noi",
            "giao ket hop dong bang loi noi",
        ),
        articles=("14", "18", "145", "162"),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=(
            "Dieu 14 hop dong lao dong bang loi noi doi voi hop dong co thoi han duoi 01 thang",
            "giao ket hop dong lao dong bang loi noi",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "noi dung chu yeu cua hop dong lao dong",
            "noi dung hop dong lao dong gom",
            "noi dung co ban cua hop dong lao dong",
            "noi dung bat buoc trong hop dong lao dong",
            "noi dung bat buoc cua hop dong lao dong",
            "nhom cac noi dung bat buoc",
        ),
        articles=("21",),
        topics=("hop_dong_lao_dong",),
        issues=("giao_ket_hop_dong",),
        expansions=("Dieu 21 noi dung hop dong lao dong",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "phu luc hop dong lao dong",
            "moi quan he giua hop dong lao dong va phu luc",
        ),
        articles=("22",),
        topics=("hop_dong_lao_dong",),
        issues=("sua_doi_bo_sung_hop_dong",),
        expansions=("Dieu 22 phu luc hop dong lao dong",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "hop dong thu viec",
            "noi dung thu viec",
            "thu viec ghi trong hop dong lao dong",
        ),
        articles=("24", "27"),
        topics=("thu_viec", "hop_dong_lao_dong"),
        issues=("thu_viec",),
        expansions=("Dieu 24 thu viec Dieu 27 ket thuc thoi gian thu viec",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "giu cccd",
            "giu can cuoc",
            "giu giay to goc",
            "giu ban chinh",
            "giu bang dai hoc",
            "giu van bang",
            "giu chung chi",
            "giay to tuy than",
        ),
        articles=("17",),
        topics=("hop_dong_lao_dong",),
        issues=("hanh_vi_cam_khi_giao_ket", "giu_giay_to_goc"),
        expansions=(
            "Dieu 17 hanh vi nguoi su dung lao dong khong duoc lam",
            "giu ban chinh giay to tuy than van bang chung chi cua nguoi lao dong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "dat coc",
            "nop tien coc",
            "bao dam bang tien",
            "bao dam bang tai san",
            "giu tien de bao dam",
        ),
        articles=("17",),
        topics=("hop_dong_lao_dong",),
        issues=("hanh_vi_cam_khi_giao_ket", "dat_coc_bao_dam"),
        expansions=(
            "Dieu 17 yeu cau nguoi lao dong thuc hien bien phap bao dam bang tien hoac tai san",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "khong tuyen phu nu",
            "phu nu co con nho",
            "co con nho",
            "loai ung vien",
        ),
        articles=("3", "8", "11", "135"),
        topics=("tuyen_dung_lao_dong", "binh_dang_phan_biet_doi_xu", "bao_ve_thai_san"),
        issues=("phan_biet_doi_xu", "tuyen_dung_lao_dong"),
        expansions=(
            "phan biet doi xu trong lao dong",
            "gioi tinh thai san trach nhiem gia dinh tinh trang HIV",
            "tuyen dung lao dong khong phan biet doi xu",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=("hiv", "nhiem hiv", "tinh trang hiv"),
        articles=("3", "8"),
        topics=("binh_dang_phan_biet_doi_xu",),
        issues=("phan_biet_doi_xu", "thong_tin_suc_khoe"),
        expansions=(
            "phan biet doi xu trong lao dong theo tinh trang HIV",
            "bao ve thong tin suc khoe trong quan he lao dong",
        ),
        confidence="low",
    ),
    RuleBasedQueryExpansion(
        phrases=("hiv", "nhiem hiv", "tinh trang hiv"),
        articles=("16", "21", "6"),
        topics=("tuyen_dung_lao_dong", "hop_dong_lao_dong"),
        issues=("du_lieu_ca_nhan", "thong_tin_suc_khoe", "thong_tin_giao_ket"),
        expansions=(
            "nghia vu cung cap thong tin khi giao ket hop dong lao dong",
            "thong tin suc khoe du lieu ca nhan trong ho so ung vien",
        ),
        context_phrases=("ho so suc khoe", "du lieu", "du lieu ca nhan", "benh an", "thong tin suc khoe"),
    ),
    RuleBasedQueryExpansion(
        phrases=("hiv", "nhiem hiv", "tinh trang hiv"),
        articles=("3", "8", "36", "37", "39"),
        topics=("binh_dang_phan_biet_doi_xu", "cham_dut_hop_dong_lao_dong"),
        issues=("phan_biet_doi_xu", "trai_phap_luat", "can_cu_cham_dut"),
        expansions=(
            "phan biet doi xu trong lao dong theo tinh trang HIV",
            "can cu nguoi su dung lao dong don phuong cham dut hop dong",
            "don phuong cham dut hop dong lao dong trai phap luat",
        ),
        context_phrases=("sa thai", "cham dut", "cho nghi", "duoi viec", "cho thoi viec"),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "sa thai mot nhan vien nu",
            "sa thai nhan vien nu",
            "sa thai vi nghi thai san",
            "sa thai vi mang thai",
            "sap nghi thai san",
        ),
        articles=("37", "3", "8", "140"),
        topics=("bao_ve_thai_san", "binh_dang_phan_biet_doi_xu", "cham_dut_hop_dong_lao_dong"),
        issues=("bao_ve_thai_san", "phan_biet_doi_xu", "trai_phap_luat"),
        expansions=(
            "khong duoc don phuong cham dut hop dong voi lao dong nu mang thai nghi thai san",
            "cam phan biet doi xu trong lao dong",
            "bao dam viec lam cho lao dong nghi thai san",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "14 tuoi",
            "tre 14 tuoi",
            "be 14 tuoi",
            "13 tuoi",
            "chua du 15 tuoi",
            "lao dong chua thanh nien",
            "hoc sinh lam them",
        ),
        articles=("143", "145", "146"),
        topics=("lao_dong_chua_thanh_nien",),
        issues=("lao_dong_chua_thanh_nien",),
        expansions=(
            "nguoi chua du 15 tuoi",
            "su dung nguoi chua du 15 tuoi",
            "thoi gio lam viec cua nguoi chua du 15 tuoi",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "lao dong nu nuoi con duoi 12 thang",
            "nuoi con duoi 12 thang",
        ),
        articles=("137", "106"),
        topics=("bao_ve_thai_san", "thoi_gio_lam_viec"),
        issues=("bao_ve_thai_san", "lam_ban_dem"),
        expansions=(
            "lao dong nu nuoi con duoi 12 thang lam viec ban dem lam them gio",
            "thoi gio lam viec vao ban dem",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "ca dem",
            "ca khuya",
            "lam ban dem",
            "lam khuya",
            "lam den nua dem",
            "nua dem",
            "qua dem",
            "22h",
            "23h",
            "2h sang",
            "5h sang",
        ),
        articles=("106",),
        topics=("thoi_gio_lam_viec",),
        issues=("lam_ban_dem",),
        expansions=("thoi gio lam viec vao ban dem tu 22 gio den 06 gio sang",),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "lao dong thue lai",
            "cho thue lai lao dong",
            "thue lai lao dong",
        ),
        articles=("52", "53", "57"),
        topics=("cho_thue_lai_lao_dong",),
        issues=("cho_thue_lai_lao_dong",),
        expansions=(
            "cho thue lai lao dong la nganh nghe kinh doanh co dieu kien",
            "khong duoc su dung lao dong thue lai de thay the nguoi lao dong dang dinh cong",
        ),
    ),
    RuleBasedQueryExpansion(
        phrases=(
            "phat tien",
            "cat luong thay viec xu ly ky luat",
            "di muon bi phat tien",
            "di muon bi tru",
            "tru tien luong vi di muon",
        ),
        articles=("127", "118", "124"),
        topics=("ky_luat_sa_thai",),
        issues=("xu_ly_ky_luat_lao_dong", "noi_quy_lao_dong"),
        expansions=(
            "Dieu 127 nghiem cam phat tien cat luong thay viec xu ly ky luat lao dong",
            "noi dung noi quy lao dong ky luat lao dong",
        ),
    ),
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
        phrases=(
            "ep viet don nghi",
            "ep viet don xin nghi",
            "buoc viet don nghi",
            "bat viet don nghi",
            "ep ky don",
            "bat toi ky don",
            "bat ky don nghi viec",
            "ky don tu nguyen nghi viec",
            "don tu nguyen nghi viec",
            "ep nghi",
            "doa cho nghi",
            "gay ap luc nghi viec",
        ),
        articles=("7", "15", "34", "36", "39", "41"),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("ep_nghi_viec", "trai_phap_luat", "boi_thuong"),
        expansions=(
            "quan he lao dong tu nguyen thien chi binh dang hop tac",
            "giao ket hop dong lao dong tu nguyen binh dang",
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
    RuleBasedQueryExpansion(
        phrases=(
            "khong co quyet dinh",
            "khong ra quyet dinh",
            "khong phai ra quyet dinh",
            "khong co giay to",
            "khong thong bao",
            "noi toi nghi roi",
        ),
        articles=("34", "36", "37", "39", "41", "48"),
        topics=("cham_dut_hop_dong_lao_dong",),
        issues=("can_cu_cham_dut", "trai_phap_luat", "nghia_vu_khi_cham_dut"),
        expansions=(
            "cac truong hop cham dut hop dong lao dong",
            "quyen don phuong cham dut hop dong lao dong cua nguoi su dung lao dong",
            "truong hop khong duoc don phuong cham dut hop dong lao dong",
            "nghia vu khi don phuong cham dut hop dong lao dong trai phap luat",
            "trach nhiem khi cham dut hop dong lao dong",
        ),
    ),
)

LEGAL_ISSUE_ARTICLE_MAP = {
    "giai_thich_tu_ngu": ("3",),
    "quyen_nghia_vu_nguoi_lao_dong": ("5",),
    "quyen_nghia_vu_nguoi_su_dung_lao_dong": ("6",),
    "hanh_vi_bi_cam": ("8",),
    "hanh_vi_cam_khi_giao_ket": ("17",),
    "giu_giay_to_goc": ("17",),
    "dat_coc_bao_dam": ("17",),
    "phan_biet_doi_xu": ("3", "8", "11", "135"),
    "quay_roi_tinh_duc": ("3", "8", "35", "118", "125"),
    "tuyen_dung_lao_dong": ("11", "16"),
    "can_cu_cham_dut": ("34", "36"),
    "quyen_don_phuong_cham_dut": ("35",),
    "thoi_han_bao_truoc": ("35", "36", "37"),
    "tro_cap_thoi_viec": ("46",),
    "tro_cap_mat_viec": ("47",),
    "nghia_vu_khi_cham_dut": ("48",),
    "trai_phap_luat": ("39", "40", "41"),
    "boi_thuong": ("40", "41", "129"),
    "sa_thai": ("124", "125"),
    "noi_quy_lao_dong": ("118", "122", "124", "125"),
    "thong_bao_cham_dut": ("35", "45"),
    "giao_ket_hop_dong": ("13", "14", "16", "17", "18", "21"),
    "sua_doi_bo_sung_hop_dong": ("22", "33"),
    "thong_tin_giao_ket": ("16",),
    "loai_hop_dong": ("20",),
    "dieu_chuyen_cong_viec": ("29",),
    "doi_thoai_tai_noi_lam_viec": ("63", "64"),
    "xu_ly_ky_luat_lao_dong": ("122", "123", "124", "127", "128", "129"),
    "tien_luong": ("94", "95", "96", "97", "98", "99", "102", "104"),
    "thoi_gio_lam_viec": ("105", "106", "107", "98"),
    "lam_ban_dem": ("106",),
    "lam_them_gio": ("107", "98"),
    "thu_viec": ("24", "25", "26", "27"),
    "bao_hiem_xa_hoi": ("48",),
    "bao_ve_thai_san": ("137", "138", "139", "140"),
    "dao_tao": ("61", "62"),
    "tam_hoan_hop_dong": ("30", "31"),
    "thay_doi_co_cau_kinh_te": ("42", "44", "47"),
    "lao_dong_chua_thanh_nien": ("143", "145", "146", "147"),
    "cho_thue_lai_lao_dong": ("52", "53", "57"),
    "tranh_chap_lao_dong": ("179", "188", "190"),
    "du_lieu_ca_nhan": ("16", "21", "6"),
    "thong_tin_suc_khoe": ("16", "21", "6"),
    "ep_nghi_viec": ("7", "15", "34", "36", "39", "41"),
    "dieu_khoan_bat_cong": ("15", "49", "51"),
    "han_che_viec_lam_sau_nghi": ("10", "21", "15"),
    "bao_mat_bi_mat_kinh_doanh": ("21", "10", "15"),
}

LEGAL_TOPIC_ARTICLE_MAP = {
    "general_provisions": ("3", "5", "6", "8"),
    "hop_dong_lao_dong": ("13", "14", "17", "20", "21", "22", "29"),
    "tro_cap": ("46", "47"),
    "bao_truoc": ("35", "36", "37"),
    "ky_luat_sa_thai": ("122", "124", "125"),
    "thay_doi_co_cau_kinh_te": ("42", "44", "47"),
    "tam_hoan_hop_dong": ("30", "31"),
    "bao_ve_thai_san": ("137", "138"),
    "tuyen_dung_lao_dong": ("11", "16"),
    "thoi_gio_lam_viec": ("105", "106", "107", "98"),
    "lao_dong_chua_thanh_nien": ("143", "145", "146", "147"),
    "cho_thue_lai_lao_dong": ("52", "53", "57"),
    "tranh_chap_lao_dong": ("179", "188", "190"),
    "binh_dang_phan_biet_doi_xu": ("3", "8", "11", "135"),
}

LEGAL_ISSUE_QUERY_HINTS = {
    "giai_thich_tu_ngu": ("giai thich tu ngu trong Bo luat Lao dong 2019",),
    "quyen_nghia_vu_nguoi_lao_dong": ("quyen va nghia vu cua nguoi lao dong",),
    "quyen_nghia_vu_nguoi_su_dung_lao_dong": ("quyen va nghia vu cua nguoi su dung lao dong",),
    "hanh_vi_bi_cam": ("cac hanh vi bi nghiem cam trong linh vuc lao dong",),
    "hanh_vi_cam_khi_giao_ket": ("hanh vi nguoi su dung lao dong khong duoc lam khi giao ket thuc hien hop dong",),
    "giu_giay_to_goc": ("giu ban chinh giay to tuy than van bang chung chi cua nguoi lao dong",),
    "dat_coc_bao_dam": ("yeu cau nguoi lao dong dat coc bao dam bang tien tai san",),
    "phan_biet_doi_xu": ("phan biet doi xu trong lao dong gioi tinh thai san HIV",),
    "quay_roi_tinh_duc": ("quay roi tinh duc tai noi lam viec",),
    "tuyen_dung_lao_dong": ("tuyen dung lao dong khong phan biet doi xu",),
    "can_cu_cham_dut": ("cac truong hop cham dut hop dong lao dong",),
    "quyen_don_phuong_cham_dut": ("quyen don phuong cham dut hop dong lao dong cua nguoi lao dong",),
    "thoi_han_bao_truoc": ("thoi han bao truoc khi don phuong cham dut hop dong lao dong",),
    "tro_cap_thoi_viec": ("tro cap thoi viec moi nam lam viec nua thang tien luong",),
    "tro_cap_mat_viec": ("tro cap mat viec it nhat bang 02 thang tien luong",),
    "nghia_vu_khi_cham_dut": ("thanh toan xac nhan thoi gian dong bao hiem xa hoi tra lai giay to",),
    "trai_phap_luat": ("don phuong cham dut hop dong lao dong trai phap luat",),
    "boi_thuong": ("boi thuong khi cham dut hop dong lao dong trai phap luat",),
    "sa_thai": ("ap dung hinh thuc xu ly ky luat sa thai",),
    "noi_quy_lao_dong": ("noi dung noi quy lao dong nguyen tac trinh tu xu ly ky luat lao dong",),
    "thong_bao_cham_dut": ("thong bao bang van ban ve viec cham dut hop dong lao dong",),
    "giao_ket_hop_dong": ("giao ket hop dong lao dong hinh thuc hop dong lao dong",),
    "loai_hop_dong": ("loai hop dong lao dong khong xac dinh thoi han xac dinh thoi han",),
    "dieu_chuyen_cong_viec": ("chuyen nguoi lao dong lam cong viec khac so voi hop dong lao dong",),
    "xu_ly_ky_luat_lao_dong": ("nguyen tac trinh tu xu ly ky luat lao dong cac hanh vi bi nghiem cam",),
    "tien_luong": ("tien luong tra luong cham lam them gio lam viec vao ban dem",),
    "thoi_gio_lam_viec": ("thoi gio lam viec binh thuong lam them gio lam viec vao ban dem",),
    "lam_ban_dem": ("thoi gio lam viec vao ban dem",),
    "lam_them_gio": ("lam them gio va tien luong lam them gio",),
    "thu_viec": ("thu viec thoi gian thu viec tien luong thu viec ket thuc thu viec",),
    "bao_ve_thai_san": ("bao ve thai san lao dong nu mang thai nuoi con duoi 12 thang",),
    "lao_dong_chua_thanh_nien": ("lao dong chua thanh nien nguoi chua du 15 tuoi lam viec ban dem",),
    "cho_thue_lai_lao_dong": ("cho thue lai lao dong khong duoc su dung lao dong thue lai",),
    "tranh_chap_lao_dong": ("tranh chap lao dong ca nhan hoa giai thoi hieu yeu cau toa an",),
    "du_lieu_ca_nhan": ("thong tin ca nhan trong giao ket va thuc hien hop dong lao dong",),
    "thong_tin_suc_khoe": ("thong tin suc khoe cua ung vien nguoi lao dong trong quan he lao dong",),
    "ep_nghi_viec": ("ep ky don nghi viec quan he lao dong tu nguyen thien chi binh dang",),
    "dieu_khoan_bat_cong": ("dieu khoan hop dong lao dong trai phap luat quyen loi thap hon quy dinh",),
    "han_che_viec_lam_sau_nghi": ("quyen lam viec tu do lua chon viec lam sau khi nghi viec",),
    "bao_mat_bi_mat_kinh_doanh": ("thoa thuan bao ve bi mat kinh doanh bi mat cong nghe trong hop dong lao dong",),
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
    force_reference_article_numbers: tuple[str, ...] = ()
    clause_refs: tuple[str, ...] = ()
    point_refs: tuple[str, ...] = ()
    query_expansions: tuple[str, ...] = ()
    query_types: tuple[str, ...] = ()

    @property
    def all_article_numbers(self) -> tuple[str, ...]:
        return dedupe_preserve_order(
            (
                *self.article_numbers,
                *self.force_reference_article_numbers,
                *self.inferred_article_numbers,
            )
        )

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


def rule_matches_normalized_query(
    normalized_query: str,
    rule: RuleBasedQueryExpansion,
) -> bool:
    if not contains_normalized_phrase(normalized_query, rule.phrases):
        return False
    if rule.context_phrases and not contains_normalized_phrase(
        normalized_query,
        rule.context_phrases,
    ):
        return False
    if rule.excluded_phrases and contains_normalized_phrase(
        normalized_query,
        rule.excluded_phrases,
    ):
        return False
    return True


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


def collect_rule_based_routing(
    normalized_query: str,
) -> RuleBasedRoutingResult:
    inferred_articles: list[str] = []
    force_reference_articles: list[str] = []
    topics: list[str] = []
    issues: list[str] = []
    expansions: list[str] = []

    for rule in (
        *LEGAL_HIGH_PRECISION_QUERY_RULES,
        *LEGAL_SOFT_HINT_QUERY_RULES,
        *TERMINATION_ARTICLE_QUERY_RULES,
    ):
        if not rule_matches_normalized_query(normalized_query, rule):
            continue
        inferred_articles.extend(rule.articles)
        if rule.confidence == "high":
            force_reference_articles.extend(rule.articles)
        topics.extend(rule.topics)
        issues.extend(rule.issues)
        expansions.extend(rule.expansions)

    return RuleBasedRoutingResult(
        inferred_articles=dedupe_preserve_order(inferred_articles),
        force_reference_articles=dedupe_preserve_order(force_reference_articles),
        topics=dedupe_preserve_order(topics),
        issues=dedupe_preserve_order(issues),
        expansions=dedupe_preserve_order(expansions),
    )


def collect_rule_based_query_expansions(
    normalized_query: str,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    routing = collect_rule_based_routing(normalized_query)
    return (
        routing.inferred_articles,
        routing.topics,
        routing.issues,
        routing.expansions,
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
    rule_routing = collect_rule_based_routing(normalized_query)
    topic_filters = dedupe_preserve_order(
        (*collect_keyword_matches(normalized_query, TOPIC_KEYWORDS), *rule_routing.topics)
    )
    issue_filters = dedupe_preserve_order(
        (*collect_keyword_matches(normalized_query, ISSUE_KEYWORDS), *rule_routing.issues)
    )
    mapped_articles, mapped_expansions = collect_mapped_article_expansions(
        topic_filters=topic_filters,
        issue_filters=issue_filters,
    )
    article_numbers = parse_reference_values(ARTICLE_REF_RE, normalized_query)
    return QueryIntent(
        raw_query=query,
        normalized_query=normalized_query,
        actor_filters=collect_keyword_matches(normalized_query, ACTOR_KEYWORDS),
        topic_filters=topic_filters,
        issue_filters=issue_filters,
        document_filters=collect_keyword_matches(normalized_query, DOCUMENT_KEYWORDS),
        article_numbers=article_numbers,
        inferred_article_numbers=dedupe_preserve_order(
            (*rule_routing.inferred_articles, *mapped_articles)
        ),
        force_reference_article_numbers=dedupe_preserve_order(
            (*article_numbers, *rule_routing.force_reference_articles)
        ),
        clause_refs=parse_reference_values(CLAUSE_REF_RE, normalized_query),
        point_refs=parse_reference_values(POINT_REF_RE, normalized_query),
        query_expansions=dedupe_preserve_order((*rule_routing.expansions, *mapped_expansions)),
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
    if intent.force_reference_article_numbers:
        parts.append(f"dieu_force={', '.join(intent.force_reference_article_numbers)}")
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
    "LEGAL_HIGH_PRECISION_QUERY_RULES",
    "LEGAL_ISSUE_ARTICLE_MAP",
    "LEGAL_ISSUE_QUERY_HINTS",
    "LEGAL_SOFT_HINT_QUERY_RULES",
    "LEGAL_TOPIC_ARTICLE_MAP",
    "MATERNITY_CONTEXT_HINTS",
    "MAX_ENUMERATION_CONTEXT_RECORDS",
    "NO_NOTICE_QUERY_HINTS",
    "QueryIntent",
    "RETIREMENT_CONTEXT_HINTS",
    "RuleBasedQueryExpansion",
    "RuleBasedRoutingResult",
    "TERMINATION_ARTICLE_MAP",
    "TERMINATION_BENEFIT_CONTEXT_HINTS",
    "TERMINATION_QUERY_HINTS",
    "TERMINATION_SECTION_HINTS",
    "YEAR_COUNT_RE",
    "build_query_variants",
    "collect_keyword_matches",
    "collect_rule_based_routing",
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
    "rule_matches_normalized_query",
]
