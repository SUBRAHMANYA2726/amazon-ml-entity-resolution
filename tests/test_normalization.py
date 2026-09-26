"""
Tests for normalization module.
"""

import pytest
import sys
sys.path.insert(0, 'src')

from amazon_ml.normalization import (
    TextNormalizer,
    BusinessNameNormalizer,
    AddressNormalizer,
    StructuredFieldNormalizer,
    Tokenizer,
    EntityNormalizer,
    NormalizedField,
    normalization_diagnostics,
)


class TestTextNormalizer:
    """Tests for TextNormalizer."""
    
    def test_unicode_normalize(self):
        tn = TextNormalizer()
        # Test NFKC normalization
        assert tn.unicode_normalize("caf\u00e9") == "caf\u00e9"  # Already composed
        assert tn.unicode_normalize("cafe\u0301") == "caf\u00e9"  # Decomposed -> composed
        # Fullwidth to halfwidth
        assert tn.unicode_normalize("\uff21\uff42\uff43") == "Abc"
    
    def test_case_normalize(self):
        tn = TextNormalizer()
        assert tn.case_normalize("HELLO") == "hello"
        assert tn.case_normalize("Hello") == "hello"
        assert tn.case_normalize("hello", lower=False) == "HELLO"
    
    def test_whitespace_normalize(self):
        tn = TextNormalizer()
        assert tn.whitespace_normalize("  hello   world  ") == "hello world"
        assert tn.whitespace_normalize("hello\tworld") == "hello world"
        assert tn.whitespace_normalize("hello\nworld") == "hello world"
        assert tn.whitespace_normalize("") == ""
    
    def test_punctuation_normalize(self):
        tn = TextNormalizer()
        # Smart quotes
        assert tn.punctuation_normalize("\u201chello\u201d") == '"hello"'
        # Em dash
        assert tn.punctuation_normalize("hello\u2014world") == "hello-world"
        # Keep specified punctuation
        assert tn.punctuation_normalize("hello, world!", keep=",") == "hello, world"
        assert tn.punctuation_normalize("A&B", keep="&") == "A&B"
    
    def test_safe_clean(self):
        tn = TextNormalizer()
        # Remove control characters
        assert tn.safe_clean("hello\x00world") == "helloworld"
        assert tn.safe_clean("hello\x1fworld") == "helloworld"
        # Keep tabs and newlines
        assert tn.safe_clean("hello\tworld") == "hello\tworld"


class TestBusinessNameNormalizer:
    """Tests for BusinessNameNormalizer."""
    
    def test_legal_suffixes(self):
        bnn = BusinessNameNormalizer()
        assert "LLC" in bnn.normalize_legal_suffixes("Acme LLC")
        assert "Inc" in bnn.normalize_legal_suffixes("Acme Inc.")
        assert "Corp" in bnn.normalize_legal_suffixes("Acme Corp")
        assert "Ltd" in bnn.normalize_legal_suffixes("Acme Ltd.")
        assert "Private Limited" in bnn.normalize_legal_suffixes("Acme Private Limited")
        assert "Pvt Ltd" in bnn.normalize_legal_suffixes("Acme Pvt Ltd")
    
    def test_ampersand(self):
        bnn = BusinessNameNormalizer()
        assert bnn.normalize_ampersand("A & B") == "A and B"
        assert bnn.normalize_ampersand("A&B") == "A and B"
    
    def test_abbreviations(self):
        bnn = BusinessNameNormalizer()
        assert "with" in bnn.normalize_abbreviations("A w/ B")
        assert "without" in bnn.normalize_abbreviations("A w/o B")
        assert "international" in bnn.normalize_abbreviations("A intl B")
    
    def test_full_normalize(self):
        bnn = BusinessNameNormalizer()
        result = bnn.full_normalize("Acme & Co. LLC")
        assert "and" in result
        assert "LLC" in result
        assert "Co" in result or "Company" in result


class TestAddressNormalizer:
    """Tests for AddressNormalizer."""
    
    def test_street_types(self):
        an = AddressNormalizer()
        assert "Street" in an.normalize_street_types("123 Main St")
        assert "Avenue" in an.normalize_street_types("123 Main Ave")
        assert "Road" in an.normalize_street_types("123 Main Rd")
        assert "Boulevard" in an.normalize_street_types("123 Main Blvd")
    
    def test_unit_types(self):
        an = AddressNormalizer()
        assert "Apt" in an.normalize_unit_types("Apt 101")
        assert "Suite" in an.normalize_unit_types("Suite 200")
        assert "Unit" in an.normalize_unit_types("# 300")
        assert "Floor" in an.normalize_unit_types("Floor 5")
    
    def test_directionals(self):
        an = AddressNormalizer()
        assert "N" in an.normalize_directionals("123 N Main St")
        assert "NE" in an.normalize_directionals("123 Northeast Main St")
    
    def test_punctuation(self):
        an = AddressNormalizer()
        assert an.normalize_punctuation("123 Main St, City, State") == "123 Main St, City, State"
        assert an.normalize_punctuation("123 Main St; City; State") == "123 Main St, City, State"
    
    def test_full_normalize(self):
        an = AddressNormalizer()
        result = an.full_normalize("123 Main St, Apt 4B, New York, NY")
        assert "Street" in result
        assert "Apt" in result


class TestStructuredFieldNormalizer:
    """Tests for StructuredFieldNormalizer."""
    
    def test_country(self):
        sfn = StructuredFieldNormalizer()
        assert sfn.normalize_country("us") == "US"
        assert sfn.normalize_country("USA") == "US"
        assert sfn.normalize_country("united states") == "US"
        assert sfn.normalize_country("india") == "India"
        assert sfn.normalize_country("france") == "France"
        # Unknown countries pass through unchanged (open-set)
        assert sfn.normalize_country("germany") == "germany"
        assert sfn.normalize_country("Germany") == "Germany"
    
    def test_city(self):
        sfn = StructuredFieldNormalizer()
        assert sfn.normalize_city("new york") == "New York"
        assert sfn.normalize_city("  london  ") == "London"
    
    def test_postal_code(self):
        sfn = StructuredFieldNormalizer()
        assert sfn.normalize_postal_code(" 12345 ") == "12345"
        assert sfn.normalize_postal_code("12345-6789") == "123456789"
        assert sfn.normalize_postal_code("SW1A 1AA") == "SW1A1AA"
    
    def test_phone(self):
        sfn = StructuredFieldNormalizer()
        assert sfn.normalize_phone("(123) 456-7890") == "1234567890"
        assert sfn.normalize_phone("+1-123-456-7890") == "11234567890"
    
    def test_email(self):
        sfn = StructuredFieldNormalizer()
        assert sfn.normalize_email("TEST@EXAMPLE.COM") == "test@example.com"
        assert sfn.normalize_email("  User@Domain.Com  ") == "user@domain.com"


class TestTokenizer:
    """Tests for Tokenizer."""
    
    def test_whitespace_tokens(self):
        tok = Tokenizer()
        assert tok.whitespace_tokens("hello world") == ["hello", "world"]
        assert tok.whitespace_tokens("  hello   world  ") == ["hello", "world"]
        assert tok.whitespace_tokens("") == []
    
    def test_alphanumeric_tokens(self):
        tok = Tokenizer()
        assert tok.alphanumeric_tokens("hello, world!") == ["hello", "world"]
        assert tok.alphanumeric_tokens("A&B Corp.") == ["A", "B", "Corp"]
        assert tok.alphanumeric_tokens("") == []
    
    def test_normalized_tokens(self):
        tok = Tokenizer()
        assert tok.normalized_tokens("Hello WORLD") == ["hello", "world"]
        assert tok.normalized_tokens("A&B Corp.") == ["a", "b", "corp"]
    
    def test_char_ngrams(self):
        tok = Tokenizer()
        assert tok.char_ngrams("hello", 3) == ["hel", "ell", "llo"]
        assert tok.char_ngrams("hi", 3) == []
        assert tok.char_ngrams("", 3) == []
    
    def test_word_ngrams(self):
        tok = Tokenizer()
        assert tok.word_ngrams(["hello", "world", "test"], 2) == ["hello world", "world test"]
        assert tok.word_ngrams(["hello"], 2) == []


class TestEntityNormalizer:
    """Tests for EntityNormalizer."""
    
    def test_normalize_business_name(self):
        en = EntityNormalizer()
        result = en.normalize_business_name("Acme & Co. LLC")
        assert result.raw == "Acme & Co. LLC"
        assert "and" in result.field_specific_normalized
        assert "LLC" in result.field_specific_normalized
        assert len(result.tokens) > 0
        assert len(result.char_ngrams) > 0
    
    def test_normalize_business_address(self):
        en = EntityNormalizer()
        result = en.normalize_business_address("123 Main St, Apt 4B")
        assert result.raw == "123 Main St, Apt 4B"
        assert "Street" in result.field_specific_normalized
        assert "Apt" in result.field_specific_normalized
    
    def test_normalize_country(self):
        en = EntityNormalizer()
        result = en.normalize_country("us")
        assert result.field_specific_normalized == "US"
    
    def test_normalize_record(self):
        en = EntityNormalizer()
        record = {
            'entity_id': 'S1-123',
            'business_name': 'Acme & Co. LLC',
            'business_address': '123 Main St, Apt 4B',
            'country': 'us',
        }
        result = en.normalize_record(record)
        assert result['entity_id'] == 'S1-123'
        assert 'business_name' in result
        assert 'business_address' in result
        assert 'country' in result
        assert result['business_name']['raw'] == 'Acme & Co. LLC'
        assert result['business_address']['raw'] == '123 Main St, Apt 4B'
        assert result['country']['raw'] == 'us'


class TestNormalizationDiagnostics:
    """Tests for normalization diagnostics."""
    
    def test_diagnostics(self):
        raw = ["Acme LLC", "Acme Inc", "Acme LLC", "Beta Corp"]
        norm = ["Acme LLC", "Acme Inc", "Acme LLC", "Beta Corp"]
        diag = normalization_diagnostics(raw, norm, "business_name")
        assert diag['total'] == 4
        assert diag['changed'] == 0
        assert diag['collision_count'] == 1  # Acme LLC appears twice
    
    def test_diagnostics_with_changes(self):
        raw = ["Acme & Co", "Acme and Co", "Beta Corp"]
        norm = ["Acme and Co", "Acme and Co", "Beta Corp"]
        diag = normalization_diagnostics(raw, norm, "business_name")
        assert diag['changed'] == 1
        assert diag['collision_count'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])