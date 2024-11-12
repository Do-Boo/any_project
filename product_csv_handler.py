import csv
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ProductCSVHandler:
    def __init__(self, filename='products.csv'):
        self.filename = filename
        self.fieldnames = [
            'category_id',            # A열: 메이크샵 카테고리
            'name',                   # D열: 상품명
            'display_name',           # E열: 모바일 상품명
            'option_mandatory',       # F열: 옵션 필수 여부
            'option_mix',             # G열: 옵션 조합 여부
            'option_name',            # H열: 옵션명
            'option_values',          # I열: 옵션값
            'option_prices',          # J열: 옵션 가격
            'opt_use',               # O열: 사용여부
            'opt_oneclick',          # P열: 옵션 한번클릭 여부
            'option_type',           # AD열: 옵션 출력 방식
            'selling_price',         # AE열: 판매가격
            'stock',                 # AF열: 재고수량
            'consumer_price',        # AG열: 소비자가격
            'supply_price',          # AH열: 상품 구매원가
            'point_percentage',      # AI열: 적립금 비율
            'reserve',               # AI열: 적립금
            'mobile_reserve',        # AJ열: 모바일 적립금
            'point',                 # AK열: 포인트
            'search_tags',           # AM열: 키워드
            'adult_auth',            # AM열: 판매가능여부
            'display_status',        # AN열: 상품진열여부
            'main_image',            # AO열: 확대이미지
            'detail_image',          # AP열: 상세이미지
            'list_image',            # AQ열: 리스트이미지
            'mobile_image',          # AR열: 모바일 이미지
            'description',           # AS열: 상품상세
            'mobile_description',    # AT열: 모바일 상세설명
            'model_name',            # BD열: 모델명
            'best_product_display',  # BE열: 베스트상품 진열여부
            'vat_type'              # BG열: 부가세 설정
        ]
        self._initialize_csv()
        
    def _initialize_csv(self):
        """CSV 파일 초기화 및 헤더 생성"""
        try:
            if not os.path.exists(self.filename):
                with open(self.filename, 'w', newline='', encoding='utf-8-sig') as f:
                    writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                    writer.writeheader()
                logger.info(f"CSV 파일 생성 완료: {self.filename}")
        except Exception as e:
            logger.error(f"CSV 파일 초기화 중 오류 발생: {str(e)}")
            raise

    def append_product(self, product_info: Dict) -> bool:
        """상품 정보를 CSV 파일에 추가"""
        try:
            # 필드 검증
            for field in self.fieldnames:
                if field not in product_info:
                    product_info[field] = ''  # 누락된 필드에 빈 값 설정
            
            with open(self.filename, 'a', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writerow(product_info)
            logger.info(f"상품 정보 추가 완료: {product_info['name']}")
            return True
        except Exception as e:
            logger.error(f"상품 정보 저장 중 오류 발생: {str(e)}")
            return False

    def get_saved_products(self) -> set:
        """저장된 상품명 목록 반환"""
        saved_products = set()
        try:
            if os.path.exists(self.filename):
                with open(self.filename, 'r', encoding='utf-8-sig') as f:
                    reader = csv.DictReader(f)
                    saved_products = {row['name'] for row in reader}
        except Exception as e:
            logger.error(f"저장된 상품 목록 조회 중 오류 발생: {str(e)}")
        return saved_products