// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title FaceProvenanceRegistry
 * @dev Immutable on-chain registry for face provenance records with event logging and tamper verification.
 */
contract FaceProvenanceRegistry {
    struct PostRecord {
        bytes32 contentHash;      // Keccak-256 digest of canonical metadata
        string sourceUrl;         // URL of the discovered social/web post
        string provider;          // Search provider identifier (e.g., serpapi_lens)
        string author;            // Discovered post author or account handle
        string postId;            // Unique post identifier
        uint256 postTimestamp;    // Extracted publication timestamp (unix epoch seconds)
        uint256 blockTimestamp;   // On-chain registration block timestamp
        address registrant;       // Account executing registration
        bool exists;              // Existence flag
    }

    // Mapping from canonical content hash to PostRecord
    mapping(bytes32 => PostRecord) private _records;

    // Ordered array of registered content hashes
    bytes32[] private _registeredHashes;

    // Events
    event PostRegistered(
        bytes32 indexed contentHash,
        string sourceUrl,
        string provider,
        string author,
        string postId,
        uint256 postTimestamp,
        uint256 registrationTimestamp,
        address indexed registrant
    );

    event PostVerified(
        bytes32 indexed contentHash,
        bool exists,
        address indexed verifier,
        uint256 timestamp
    );

    // Custom Errors
    error InvalidContentHash();
    error RecordAlreadyExists(bytes32 contentHash);
    error RecordNotFound(bytes32 contentHash);

    /**
     * @notice Registers a new face provenance record.
     * @param contentHash Keccak-256 digest of the canonical metadata JSON
     * @param sourceUrl Discovered URL of the post
     * @param provider Search provider identifier
     * @param author Post author or account handle
     * @param postId Unique post identifier
     * @param postTimestamp Unix timestamp of post publication
     * @return bool Returns true upon successful registration
     */
    function registerPost(
        bytes32 contentHash,
        string calldata sourceUrl,
        string calldata provider,
        string calldata author,
        string calldata postId,
        uint256 postTimestamp
    ) external returns (bool) {
        if (contentHash == bytes32(0)) {
            revert InvalidContentHash();
        }
        if (_records[contentHash].exists) {
            revert RecordAlreadyExists(contentHash);
        }

        PostRecord memory record = PostRecord({
            contentHash: contentHash,
            sourceUrl: sourceUrl,
            provider: provider,
            author: author,
            postId: postId,
            postTimestamp: postTimestamp,
            blockTimestamp: block.timestamp,
            registrant: msg.sender,
            exists: true
        });

        _records[contentHash] = record;
        _registeredHashes.push(contentHash);

        emit PostRegistered(
            contentHash,
            sourceUrl,
            provider,
            author,
            postId,
            postTimestamp,
            block.timestamp,
            msg.sender
        );

        return true;
    }

    /**
     * @notice Retrieves the full PostRecord for a registered content hash.
     * @param contentHash Keccak-256 digest
     * @return PostRecord memory struct
     */
    function getPost(bytes32 contentHash) external view returns (PostRecord memory) {
        if (!_records[contentHash].exists) {
            revert RecordNotFound(contentHash);
        }
        return _records[contentHash];
    }

    /**
     * @notice Verifies whether a content hash is registered and emits a PostVerified event.
     * @param contentHash Keccak-256 digest to verify
     * @return exists True if record exists on-chain
     * @return registrationTimestamp Block timestamp of initial registration (or 0)
     * @return sourceUrl Source URL stored on-chain (or empty)
     */
    function verifyPost(bytes32 contentHash) external returns (bool exists, uint256 registrationTimestamp, string memory sourceUrl) {
        bool isFound = _records[contentHash].exists;
        uint256 regTime = isFound ? _records[contentHash].blockTimestamp : 0;
        string memory src = isFound ? _records[contentHash].sourceUrl : "";

        emit PostVerified(contentHash, isFound, msg.sender, block.timestamp);
        return (isFound, regTime, src);
    }

    /**
     * @notice Read-only check for whether a post hash is registered.
     * @param contentHash Keccak-256 digest
     * @return bool True if registered
     */
    function isRegistered(bytes32 contentHash) external view returns (bool) {
        return _records[contentHash].exists;
    }

    /**
     * @notice Alias for isRegistered.
     */
    function isPostRegistered(bytes32 contentHash) external view returns (bool) {
        return _records[contentHash].exists;
    }

    /**
     * @notice Returns total number of registered records.
     */
    function totalRecords() external view returns (uint256) {
        return _registeredHashes.length;
    }

    /**
     * @notice Alias for totalRecords.
     */
    function getTotalPosts() external view returns (uint256) {
        return _registeredHashes.length;
    }

    /**
     * @notice Returns all registered content hashes.
     */
    function getRegisteredHashes() external view returns (bytes32[] memory) {
        return _registeredHashes;
    }

    /**
     * @notice Returns a record by its insertion index.
     */
    function getRecordByIndex(uint256 index) external view returns (PostRecord memory) {
        require(index < _registeredHashes.length, "Index out of bounds");
        bytes32 hashVal = _registeredHashes[index];
        return _records[hashVal];
    }
}
