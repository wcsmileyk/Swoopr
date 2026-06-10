"""
Payment service — provider-agnostic payment processing.

StubProvider is the active default and performs no real transactions.
To add a real provider:
  1. Implement a subclass of PaymentProvider
  2. Register it in _PROVIDERS
  3. Set payment.provider = '<key>' before calling capture_payment

Public API
----------
create_payment(reservation, provider='stub') -> Payment
capture_payment(payment, card_token=None)    -> bool
refund_payment(payment)                      -> bool
void_payment(payment)                        -> bool
"""

import abc
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------------

class PaymentProvider(abc.ABC):
    @abc.abstractmethod
    def charge(self, payment, card_token=None):
        """Attempt to capture. Returns (success, transaction_id, data)."""

    @abc.abstractmethod
    def refund(self, payment):
        """Refund a captured payment. Returns (success, data)."""

    @abc.abstractmethod
    def void(self, payment):
        """Void an authorized/pending payment. Returns (success, data)."""


class StubProvider(PaymentProvider):
    """No-op provider — always succeeds, no real transaction."""

    def charge(self, payment, card_token=None):
        logger.info('StubProvider.charge payment=%s amount=%s', payment.pk, payment.amount_cents)
        return True, f'stub-{payment.pk}', {'provider': 'stub', 'note': 'no real transaction'}

    def refund(self, payment):
        logger.info('StubProvider.refund payment=%s', payment.pk)
        return True, {'provider': 'stub', 'note': 'no real transaction'}

    def void(self, payment):
        logger.info('StubProvider.void payment=%s', payment.pk)
        return True, {'provider': 'stub', 'note': 'no real transaction'}


_PROVIDERS = {
    'stub': StubProvider,
    # 'stripe': StripeProvider,   # plug in when ready
    # 'square': SquareProvider,
    # 'cash':   CashProvider,
}


def _get_provider(provider_name):
    cls = _PROVIDERS.get(provider_name, StubProvider)
    return cls()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_payment(reservation, provider='stub'):
    """
    Create a Payment record for a reservation in pending status.
    Returns existing Payment if one already exists.
    """
    from dz.models import Payment

    if hasattr(reservation, 'payment'):
        return reservation.payment

    return Payment.objects.create(
        reservation=reservation,
        provider=provider,
        status='pending',
        amount_cents=reservation.quoted_price_cents or 0,
        currency='USD',
    )


def capture_payment(payment, card_token=None):
    """
    Attempt to capture a payment. Updates payment.status and saves.
    Returns True on success.
    """
    provider = _get_provider(payment.provider)
    success, transaction_id, data = provider.charge(payment, card_token=card_token)
    if success:
        payment.status = 'captured'
        payment.provider_transaction_id = transaction_id
        payment.provider_data = data
    else:
        payment.status = 'failed'
        payment.provider_data = data
    payment.save()
    return success


def refund_payment(payment):
    """
    Refund a captured payment. Returns True on success.
    """
    if payment.status != 'captured':
        return False
    provider = _get_provider(payment.provider)
    success, data = provider.refund(payment)
    if success:
        payment.status = 'refunded'
        payment.provider_data = {**payment.provider_data, 'refund': data}
        payment.save()
    return success


def void_payment(payment):
    """
    Void a pending or authorized payment. Returns True on success.
    """
    if payment.status not in ('pending', 'authorized'):
        return False
    provider = _get_provider(payment.provider)
    success, data = provider.void(payment)
    if success:
        payment.status = 'voided'
        payment.provider_data = {**payment.provider_data, 'void': data}
        payment.save()
    return success
