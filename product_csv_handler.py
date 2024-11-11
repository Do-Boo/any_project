import csv
import os
import logging
from typing import Dict

logger = logging.getLogger(__name__)

class ProductCSVHandler:
    def __init__(self, filename='products.csv'):
        self.filename = filename
        self.fieldnames = [
            'category_id',
            'name',
            'display_name',
            'option_type',
            'selling_price',
            'supply_price',
            'consumer_price',
            'min_quantity',
            'point_percentage',
            'point_amount',
            'additional_point',
            'search_tags',
            'adult_auth',
            'display_status',
            'main_image',
            'detail_image',
            'list_image',
            'description',
            'phone_model'
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