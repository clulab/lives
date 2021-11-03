"""LIvES Data Wrangling Exceptions"""

class AnnotationBaseError(Exception):
    def __init__(self, item, item_type, annotation_dict):
        self.item:str = item
        self.item_type:str = item_type
        self.annotation_dict:dict = annotation_dict
        self.message = f"This annotation's {item_type} is unknown"
        super().__init__(self.message)

    def __str__(self):
        return f'*** Bad item: {self.annotation_dict} -> {self.message}\n*** See: {self.item} '


class AnnotationCheckError(AnnotationBaseError):
    def __init__(self, item, item_type, annotation_dict):
        self.item:str = item
        self.item_type:str = item_type
        self.annotation_dict = annotation_dict
        self.message = f"*** The following annotation will not appear in the new codebook: `{item_type}` ***"
