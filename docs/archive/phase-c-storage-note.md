# Phase C validation storage incident

Development source-fusion results were already frozen before this incident.
No validation ranking or score was available during policy selection.

Original validation downloads were saved to the user-approved external folder.
Its ExFAT allocation size made 57,638 FiQA text files occupy approximately 56 GiB.
An independent 150 GiB APFS sparse image was created *as a file* in that folder,
mounted at `/Volumes/ARKBPhaseC`. This did not format the external drive. All three
full corpora were normalized and materialized in the image. NFCorpus and FiQA
normalized file hashes matched the first materialization exactly, and their new
materialized text was verified before removing only those redundant task copies.
Raw downloads and every prior P4/Phase A/Phase B artifact remain intact.

Qdrant 1.19.0 host-bind storage on OrbStack reported an incompatible FUSE filesystem.
Both the existing server and a fresh isolated host-bind server stalled while
creating a collection. The old server was neither restarted nor reconfigured.
Following [Qdrant's official filesystem guidance](https://qdrant.tech/documentation/guides/common-errors/),
a separate native Docker volume `arkb-phase-c-native-v1` was provisioned with the
same Qdrant 1.19.0 image and original index parameters. Its service runs at
`http://127.0.0.1:6340`, exposed only on localhost. The existing real-server
publication/HNSW/failure-recovery test passed in 3.54 seconds on this instance.
Thus the change is operational storage isolation, not a retrieval-policy or
Qdrant index-configuration experiment. The user instruction to freeze Qdrant
retrieval configuration remains satisfied.

The two NFCorpus attempts that stopped before retrieval are preserved separately.
The successful NFCorpus run used the same production builder with 3,887 cached
embedding inputs: it embedded zero additional inputs and produced all 3,930 chunks
from all 3,633 documents. Its 323 queries were retrieved once. No failed attempt
produced rankings, and none was scored or used to tune a policy.

Embedding-cache staging reuses production scanning, chunking, input preparation,
tokenization, embedding requests and immutable SQLite cache keys. It neither
constructs a fake index nor publishes a snapshot. The normal builder remains
responsible for identity validation, Qdrant verification and atomic publication.
The staging source and input/model identities are separately hashed.

Large SQLite/corpus/artifact files are kept on the external APFS image. Active
Qdrant vectors use the native Linux Docker volume on the system disk; free space
must be monitored, and snapshots must be retained externally for Phase E. This
also avoids binding an external macOS volume directly into Qdrant.
