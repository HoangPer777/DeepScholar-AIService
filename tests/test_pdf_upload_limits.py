import os

import pytest

os.environ.setdefault("INTERNAL_SERVICE_KEY", "test-internal-key")

from app.api import pdf


class FakeBody:
    def __init__(self, data):
        self.data = data
        self.closed = False

    def read(self, _size):
        if self.data is None:
            return b''
        data, self.data = self.data, None
        return data

    def close(self):
        self.closed = True


class FakeS3:
    def __init__(self, content_length, body_data):
        self.content_length = content_length
        self.body = FakeBody(body_data)
        self.deleted = False
        self.get_called = False

    def head_object(self, **_kwargs):
        return {'ContentLength': self.content_length}

    def get_object(self, **_kwargs):
        self.get_called = True
        return {'Body': self.body}

    def delete_object(self, **_kwargs):
        self.deleted = True


def test_head_metadata_rejects_and_cleans_oversized_object(monkeypatch):
    fake_s3 = FakeS3(pdf.MAX_PDF_SIZE_BYTES + 1, b'not-read')
    monkeypatch.setattr(pdf, 'get_s3_client', lambda: fake_s3)

    with pytest.raises(ValueError, match='25 MiB'):
        pdf._download_pdf_from_r2('https://files.example/articles/large.pdf')

    assert fake_s3.deleted is True
    assert fake_s3.get_called is False


def test_streaming_cap_rejects_when_metadata_underreports_size(monkeypatch):
    fake_s3 = FakeS3(1, b'x' * (pdf.MAX_PDF_SIZE_BYTES + 1))
    monkeypatch.setattr(pdf, 'get_s3_client', lambda: fake_s3)

    with pytest.raises(ValueError, match='25 MiB'):
        pdf._download_pdf_from_r2('https://files.example/articles/underreported.pdf')

    assert fake_s3.deleted is True
    assert fake_s3.body.closed is True


def test_download_accepts_pdf_at_or_below_limit(monkeypatch):
    fake_s3 = FakeS3(3, b'pdf')
    monkeypatch.setattr(pdf, 'get_s3_client', lambda: fake_s3)

    result = pdf._download_pdf_from_r2('https://files.example/articles/valid.pdf')

    assert result == b'pdf'
    assert fake_s3.deleted is False
    assert fake_s3.body.closed is True
