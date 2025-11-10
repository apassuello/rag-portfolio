
# Cache API Status Code Fix

The cache API tests are failing because they expect validation errors (422) 
but the service returns 404 for all invalid hashes.

## Issue Analysis:
1. test_cache_get_endpoint_invalid_hash expects 422 but gets 404
2. test_cache_post_endpoint_basic may have similar issues
3. test_cache_post_then_get_workflow may have response format issues
4. test_cache_delete_endpoint may have response format issues

## TDD Fix Approach:
1. First understand what the actual API returns
2. Write tests that match the actual behavior  
3. Fix tests to have correct expectations
4. Ensure test logic is sound

## Status Code Standards:
- 404: Resource not found (this is what cache service returns for invalid/missing hashes)
- 422: Validation error (this is what we expected, but service uses 404)
- 200: Success
- 500: Server error (should be avoided)

## Recommended Fixes:
1. Change expectations from 422 to 404 for invalid hashes
2. Update response format expectations to match actual service
3. Ensure proper JSON parsing and field validation
