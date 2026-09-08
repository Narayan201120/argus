"""P7-1 isolation matrix skeleton (DEC-056).

P7-0 ships enforcement plumbing only, so every case here is skipped.
P7-1 un-skips them as routes take the server subject and return 404 on
cross-user access. No id oracle, no board migration.
"""

import pytest

CASES = [
    ("alice", "bob", "read other investigation", 404),
    ("alice", "bob", "cancel other investigation", 404),
    ("alice", "bob", "rate other investigation", 404),
    ("alice", None, "unauthenticated read", 401),
    ("alice", None, "unauthenticated cancel", 401),
    ("legacy-local", "alice", "legacy row invisible to new subject", 404),
]


@pytest.mark.skip(reason="P7-1 cutover: per-user scoping not enforced yet")
@pytest.mark.parametrize("owner,caller,action,expected", CASES)
def test_isolation_matrix(owner, caller, action, expected):
    raise NotImplementedError(action)
