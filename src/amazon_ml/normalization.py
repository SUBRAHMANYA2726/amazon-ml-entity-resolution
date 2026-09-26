"""
Normalization module for Amazon ML Entity Resolution.
Preserves raw values and adds normalized representations side-by-side.
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class NormalizedField:
    """Container for raw and normalized field representations."""
    raw: str
    unicode_normalized: str = ""
    case_whitespace_normalized: str = ""
    punctuation_normalized: str = ""
    field_specific_normalized: str = ""
    tokens: List[str] = field(default_factory=list)
    char_ngrams: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'raw': self.raw,
            'unicode_normalized': self.unicode_normalized,
            'case_whitespace_normalized': self.case_whitespace_normalized,
            'punctuation_normalized': self.punctuation_normalized,
            'field_specific_normalized': self.field_specific_normalized,
            'tokens': self.tokens,
            'char_ngrams': self.char_ngrams,
        }


class TextNormalizer:
    """Conservative text normalization preserving information."""
    
    @staticmethod
    def unicode_normalize(text: str, form: str = 'NFKC') -> str:
        """Apply Unicode normalization (NFKC by default)."""
        if not text:
            return ""
        return unicodedata.normalize(form, text)
    
    @staticmethod
    def case_normalize(text: str, lower: bool = True) -> str:
        """Apply case folding."""
        if not text:
            return ""
        return text.lower() if lower else text.upper()
    
    @staticmethod
    def whitespace_normalize(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces, strip, remove tabs/newlines."""
        if not text:
            return ""
        # Replace tabs, newlines with space
        text = re.sub(r'[\t\n\r\f\v]', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' +', ' ', text)
        # Strip leading/trailing
        return text.strip()
    
    @staticmethod
    def punctuation_normalize(text: str, keep: str = ".,-&'\"") -> str:
        """
        Normalize punctuation: standardize similar punctuation, 
        optionally keep certain characters.
        """
        if not text:
            return ""
        # Standardize unicode punctuation to ASCII equivalents
        replacements = {
            '\u2018': "'",  # left single quote
            '\u2019': "'",  # right single quote
            '\u201c': '"',  # left double quote
            '\u201d': '"',  # right double quote
            '\u2013': '-',  # en dash
            '\u2014': '-',  # em dash
            '\u2026': '...',  # ellipsis
            '\u00a0': ' ',  # non-breaking space
            '\u200b': '',   # zero-width space
        }
        for uni, ascii_repl in replacements.items():
            text = text.replace(uni, ascii_repl)
        
        # Remove punctuation not in keep list (replace with space)
        if keep:
            keep_pattern = re.escape(keep)
            text = re.sub(f'[^\\w\\s{keep_pattern}]', ' ', text)
        
        # Clean up any resulting double spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    @staticmethod
    def safe_clean(text: str) -> str:
        """Remove control characters and other problematic characters."""
        if not text:
            return ""
        # Remove control characters except tab, newline (already handled)
        text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in '\t\n')
        return text


class BusinessNameNormalizer:
    """Business name normalization based on observed dataset patterns."""
    
    # Legal suffixes observed in the dataset
    LEGAL_SUFFIXES = {
        'llc': 'LLC',
        'inc': 'Inc',
        'corp': 'Corp',
        'ltd': 'Ltd',
        'llp': 'LLP',
        'lp': 'LP',
        'pc': 'PC',
        'pllc': 'PLLC',
        'private limited': 'Private Limited',
        'pvt ltd': 'Pvt Ltd',
        'pvt. ltd': 'Pvt Ltd',
        'pvt ltd.': 'Pvt Ltd',
        'co': 'Co',
        'company': 'Company',
        'corporation': 'Corporation',
        'incorporated': 'Incorporated',
        'limited': 'Limited',
    }
    
    # Common abbreviations
    ABBREVIATIONS = {
        '&': 'and',
        'w/': 'with',
        'w/o': 'without',
        'vs': 'versus',
        'dept': 'department',
        'div': 'division',
        'intl': 'international',
        'mfg': 'manufacturing',
        'svcs': 'services',
        'solns': 'solutions',
        'tech': 'technology',
        'mgmt': 'management',
        'admin': 'administration',
        'assoc': 'associates',
        'grp': 'group',
        'hdqtrs': 'headquarters',
        'hq': 'headquarters',
    }
    
    @classmethod
    def normalize_legal_suffixes(cls, text: str) -> str:
        """Standardize legal suffixes to canonical forms."""
        if not text:
            return ""
        # Sort by length descending to match longer patterns first
        sorted_suffixes = sorted(cls.LEGAL_SUFFIXES.keys(), key=len, reverse=True)
        for suffix in sorted_suffixes:
            pattern = rf'\b{re.escape(suffix)}\.?\b'
            replacement = f' {cls.LEGAL_SUFFIXES[suffix]} '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r' +', ' ', text).strip()
    
    @classmethod
    def normalize_ampersand(cls, text: str) -> str:
        """Normalize '&' to 'and'."""
        if not text:
            return ""
        # Replace & with and, handling surrounding spaces
        text = re.sub(r'\s*&\s*', ' and ', text)
        return text
    
    @classmethod
    def normalize_abbreviations(cls, text: str) -> str:
        """Expand common abbreviations."""
        if not text:
            return ""
        # Sort by length descending
        sorted_abbrevs = sorted(cls.ABBREVIATIONS.items(), key=lambda x: len(x[0]), reverse=True)
        for abbrev, expansion in sorted_abbrevs:
            # Use word boundary or non-word char for patterns with special chars
            if re.search(r'\W', abbrev):
                pattern = rf'(?<!\w){re.escape(abbrev)}(?!\w)'
            else:
                pattern = rf'\b{re.escape(abbrev)}\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    @classmethod
    def normalize_spacing(cls, text: str) -> str:
        """Normalize spacing around punctuation and suffixes."""
        if not text:
            return ""
        # Space around commas, periods
        text = re.sub(r'\s*,\s*', ', ', text)
        text = re.sub(r'\s*\.\s*', '. ', text)
        # Remove double spaces
        text = re.sub(r' +', ' ', text)
        return text.strip()
    
    @classmethod
    def full_normalize(cls, text: str) -> str:
        """Apply full business name normalization pipeline."""
        if not text:
            return ""
        text = cls.normalize_ampersand(text)
        text = cls.normalize_legal_suffixes(text)
        text = cls.normalize_abbreviations(text)
        text = cls.normalize_spacing(text)
        return text


class AddressNormalizer:
    """Address normalization based on observed dataset patterns."""
    
    STREET_TYPES = {
        'st': 'Street',
        'street': 'Street',
        'ave': 'Avenue',
        'avenue': 'Avenue',
        'rd': 'Road',
        'road': 'Road',
        'blvd': 'Boulevard',
        'boulevard': 'Boulevard',
        'dr': 'Drive',
        'drive': 'Drive',
        'ln': 'Lane',
        'lane': 'Lane',
        'ct': 'Court',
        'court': 'Court',
        'pl': 'Place',
        'place': 'Place',
        'pkwy': 'Parkway',
        'parkway': 'Parkway',
        'cir': 'Circle',
        'circle': 'Circle',
        'tr': 'Trail',
        'trail': 'Trail',
        'way': 'Way',
        'hwy': 'Highway',
        'highway': 'Highway',
    }
    
    UNIT_TYPES = {
        'apt': 'Apt',
        'apartment': 'Apt',
        'suite': 'Suite',
        'ste': 'Suite',
        'unit': 'Unit',
        'floor': 'Floor',
        'fl': 'Floor',
        'bldg': 'Building',
        'building': 'Building',
        'room': 'Room',
        'rm': 'Room',
        '#': 'Unit',
    }
    
    DIRECTIONALS = {
        'n': 'N',
        's': 'S',
        'e': 'E',
        'w': 'W',
        'ne': 'NE',
        'nw': 'NW',
        'se': 'SE',
        'sw': 'SW',
        'north': 'N',
        'south': 'S',
        'east': 'E',
        'west': 'W',
        'northeast': 'NE',
        'northwest': 'NW',
        'southeast': 'SE',
        'southwest': 'SW',
    }
    
    @classmethod
    def normalize_street_types(cls, text: str) -> str:
        """Standardize street type abbreviations."""
        if not text:
            return ""
        sorted_types = sorted(cls.STREET_TYPES.keys(), key=len, reverse=True)
        for st_type in sorted_types:
            pattern = rf'\b{re.escape(st_type)}\.?\b'
            replacement = f' {cls.STREET_TYPES[st_type]} '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r' +', ' ', text).strip()
    
    @classmethod
    def normalize_unit_types(cls, text: str) -> str:
        """Standardize unit/apartment/suite abbreviations."""
        if not text:
            return ""
        sorted_types = sorted(cls.UNIT_TYPES.keys(), key=len, reverse=True)
        for unit_type in sorted_types:
            pattern = rf'\b{re.escape(unit_type)}\.?\b'
            replacement = f' {cls.UNIT_TYPES[unit_type]} '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        # Handle # prefix
        text = re.sub(r'#\s*(\d+)', r'Unit \1', text)
        return re.sub(r' +', ' ', text).strip()
    
    @classmethod
    def normalize_directionals(cls, text: str) -> str:
        """Standardize directional prefixes/suffixes."""
        if not text:
            return ""
        sorted_dirs = sorted(cls.DIRECTIONALS.keys(), key=len, reverse=True)
        for direction in sorted_dirs:
            pattern = rf'\b{re.escape(direction)}\.?\b'
            replacement = f' {cls.DIRECTIONALS[direction]} '
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        return re.sub(r' +', ' ', text).strip()
    
    @classmethod
    def normalize_punctuation(cls, text: str) -> str:
        """Normalize address punctuation."""
        if not text:
            return ""
        # Normalize commas
        text = re.sub(r'\s*,\s*', ', ', text)
        # Normalize semicolons to commas
        text = re.sub(r'\s*;\s*', ', ', text)
        # Normalize periods
        text = re.sub(r'\s*\.\s*', ' ', text)
        # Handle fractions like "19 1/2"
        text = re.sub(r'(\d+)\s+(\d)/(\d)', r'\1 \2/\3', text)
        return re.sub(r' +', ' ', text).strip()
    
    @classmethod
    def full_normalize(cls, text: str) -> str:
        """Apply full address normalization pipeline."""
        if not text:
            return ""
        text = cls.normalize_punctuation(text)
        text = cls.normalize_street_types(text)
        text = cls.normalize_unit_types(text)
        text = cls.normalize_directionals(text)
        return text


class StructuredFieldNormalizer:
    """Normalization for structured fields (country, city, postal, phone, email)."""
    
    @staticmethod
    def normalize_country(text: str) -> str:
        """Normalize country names - keep open-set."""
        if not text:
            return ""
        text = text.strip()
        # Common variations
        country_map = {
            'usa': 'US',
            'us': 'US',
            'u.s.': 'US',
            'u.s.a.': 'US',
            'united states': 'US',
            'united states of america': 'US',
            'india': 'India',
            'france': 'France',
        }
        lower = text.lower()
        return country_map.get(lower, text)
    
    @staticmethod
    def normalize_city(text: str) -> str:
        """Normalize city names - keep open-set."""
        if not text:
            return ""
        return text.strip().title()
    
    @staticmethod
    def normalize_postal_code(text: str) -> str:
        """Normalize postal/ZIP codes."""
        if not text:
            return ""
        # Remove spaces, make uppercase
        text = text.replace(' ', '').replace('-', '').upper()
        # US ZIP: 5 digits or 5+4
        # India PIN: 6 digits
        # Generic: alphanumeric
        return text
    
    @staticmethod
    def normalize_phone(text: str) -> str:
        """Normalize phone numbers to digits only."""
        if not text:
            return ""
        # Extract digits
        digits = re.sub(r'\D', '', text)
        return digits
    
    @staticmethod
    def normalize_email(text: str) -> str:
        """Normalize email addresses."""
        if not text:
            return ""
        return text.strip().lower()


class Tokenizer:
    """Tokenization utilities."""
    
    @staticmethod
    def whitespace_tokens(text: str) -> List[str]:
        """Split on whitespace."""
        if not text:
            return []
        return text.split()
    
    @staticmethod
    def alphanumeric_tokens(text: str) -> List[str]:
        """Extract alphanumeric tokens."""
        if not text:
            return []
        return re.findall(r'[a-zA-Z0-9]+', text)
    
    @staticmethod
    def normalized_tokens(text: str) -> List[str]:
        """Lowercase alphanumeric tokens."""
        if not text:
            return []
        return [t.lower() for t in re.findall(r'[a-zA-Z0-9]+', text)]
    
    @staticmethod
    def char_ngrams(text: str, n: int = 3) -> List[str]:
        """Generate character n-grams."""
        if not text or len(text) < n:
            return []
        text = text.lower().replace(' ', '')
        return [text[i:i+n] for i in range(len(text) - n + 1)]
    
    @staticmethod
    def word_ngrams(tokens: List[str], n: int = 2) -> List[str]:
        """Generate word n-grams from token list."""
        if len(tokens) < n:
            return []
        return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]


class EntityNormalizer:
    """Main normalizer for entity records."""
    
    def __init__(self):
        self.text_normalizer = TextNormalizer()
        self.name_normalizer = BusinessNameNormalizer()
        self.addr_normalizer = AddressNormalizer()
        self.structured_normalizer = StructuredFieldNormalizer()
        self.tokenizer = Tokenizer()
    
    def normalize_business_name(self, raw: str) -> NormalizedField:
        """Normalize business name field."""
        nf = NormalizedField(raw=raw)
        
        if not raw:
            return nf
        
        # Unicode normalization
        nf.unicode_normalized = self.text_normalizer.unicode_normalize(raw)
        
        # Case + whitespace
        cw = self.text_normalizer.case_normalize(nf.unicode_normalized)
        cw = self.text_normalizer.whitespace_normalize(cw)
        nf.case_whitespace_normalized = cw
        
        # Punctuation
        pn = self.text_normalizer.punctuation_normalize(cw, keep=".,-'&")
        nf.punctuation_normalized = pn
        
        # Field-specific (business name)
        fs = self.name_normalizer.full_normalize(pn)
        nf.field_specific_normalized = fs
        
        # Tokens
        nf.tokens = self.tokenizer.normalized_tokens(fs)
        nf.char_ngrams = self.tokenizer.char_ngrams(fs)
        
        return nf
    
    def normalize_business_address(self, raw: str) -> NormalizedField:
        """Normalize business address field."""
        nf = NormalizedField(raw=raw)
        
        if not raw:
            return nf
        
        nf.unicode_normalized = self.text_normalizer.unicode_normalize(raw)
        
        cw = self.text_normalizer.case_normalize(nf.unicode_normalized)
        cw = self.text_normalizer.whitespace_normalize(cw)
        nf.case_whitespace_normalized = cw
        
        pn = self.text_normalizer.punctuation_normalize(cw, keep=".,-#/")
        nf.punctuation_normalized = pn
        
        fs = self.addr_normalizer.full_normalize(pn)
        nf.field_specific_normalized = fs
        
        nf.tokens = self.tokenizer.normalized_tokens(fs)
        nf.char_ngrams = self.tokenizer.char_ngrams(fs)
        
        return nf
    
    def normalize_country(self, raw: str) -> NormalizedField:
        """Normalize country field."""
        nf = NormalizedField(raw=raw)
        
        if not raw:
            return nf
        
        nf.unicode_normalized = self.text_normalizer.unicode_normalize(raw)
        nf.case_whitespace_normalized = self.text_normalizer.whitespace_normalize(nf.unicode_normalized)
        nf.punctuation_normalized = self.text_normalizer.punctuation_normalize(nf.case_whitespace_normalized)
        nf.field_specific_normalized = self.structured_normalizer.normalize_country(nf.punctuation_normalized)
        nf.tokens = self.tokenizer.normalized_tokens(nf.field_specific_normalized)
        
        return nf
    
    def normalize_record(self, record: Dict) -> Dict:
        """Normalize all fields in a record, preserving raw values."""
        normalized = {'entity_id': record.get('entity_id', '')}
        
        # Business name
        name_raw = record.get('business_name', '')
        normalized['business_name'] = self.normalize_business_name(name_raw).to_dict()
        
        # Business address
        addr_raw = record.get('business_address', '')
        normalized['business_address'] = self.normalize_business_address(addr_raw).to_dict()
        
        # Country
        country_raw = record.get('country', '')
        normalized['country'] = self.normalize_country(country_raw).to_dict()
        
        return normalized


def normalize_dataset(df, id_col='entity_id') -> List[Dict]:
    """Normalize entire dataset."""
    normalizer = EntityNormalizer()
    results = []
    for _, row in df.iterrows():
        record = row.to_dict()
        normalized = normalizer.normalize_record(record)
        results.append(normalized)
    return results


# Diagnostics
def normalization_diagnostics(raw_values: List[str], normalized_values: List[str], 
                               field_name: str) -> Dict:
    """Generate diagnostics for normalization."""
    if len(raw_values) != len(normalized_values):
        return {'error': 'Length mismatch'}
    
    changed = 0
    empty_norm = 0
    token_counts = []
    collisions = {}
    
    for raw, norm in zip(raw_values, normalized_values):
        if raw != norm:
            changed += 1
        if not norm:
            empty_norm += 1
        tokens = norm.split() if norm else []
        token_counts.append(len(tokens))
        # Check for collisions (different raw -> same normalized)
        if norm in collisions:
            collisions[norm].append(raw)
        else:
            collisions[norm] = [raw]
    
    # Filter to only actual collisions
    collisions = {k: v for k, v in collisions.items() if len(v) > 1}
    
    return {
        'field': field_name,
        'total': len(raw_values),
        'changed': changed,
        'changed_pct': round(changed / len(raw_values) * 100, 2) if raw_values else 0,
        'empty_normalized': empty_norm,
        'empty_normalized_pct': round(empty_norm / len(raw_values) * 100, 2) if raw_values else 0,
        'avg_tokens': round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        'max_tokens': max(token_counts) if token_counts else 0,
        'min_tokens': min(token_counts) if token_counts else 0,
        'collision_count': len(collisions),
        'example_collisions': dict(list(collisions.items())[:5]),
    }