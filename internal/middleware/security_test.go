package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestRateLimiter_Allow(t *testing.T) {
	limiter := NewRateLimiter(5, time.Minute)

	// Should allow first 5 requests
	for i := 0; i < 5; i++ {
		if !limiter.Allow("127.0.0.1") {
			t.Errorf("Request %d should be allowed", i+1)
		}
	}

	// 6th request should be blocked
	if limiter.Allow("127.0.0.1") {
		t.Error("6th request should be blocked")
	}

	// Different IP should be allowed
	if !limiter.Allow("192.168.1.1") {
		t.Error("Different IP should be allowed")
	}
}

func TestRateLimit_Middleware(t *testing.T) {
	limiter := NewRateLimiter(2, time.Minute)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := RateLimit(limiter)(handler)

	// First 2 requests should succeed
	for i := 0; i < 2; i++ {
		req := httptest.NewRequest("GET", "/test", nil)
		req.RemoteAddr = "127.0.0.1:12345"
		rr := httptest.NewRecorder()

		middleware.ServeHTTP(rr, req)

		if rr.Code != http.StatusOK {
			t.Errorf("Request %d: expected status %d, got %d", i+1, http.StatusOK, rr.Code)
		}
	}

	// 3rd request should be rate limited
	req := httptest.NewRequest("GET", "/test", nil)
	req.RemoteAddr = "127.0.0.1:12345"
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Code != http.StatusTooManyRequests {
		t.Errorf("Expected status %d, got %d", http.StatusTooManyRequests, rr.Code)
	}
}

func TestRateLimit_XForwardedFor(t *testing.T) {
	limiter := NewRateLimiter(1, time.Minute)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := RateLimit(limiter)(handler)

	// Request with X-Forwarded-For
	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("X-Forwarded-For", "10.0.0.1, 10.0.0.2")
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, rr.Code)
	}
}

func TestCORS_AllowedOrigin(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := CORS([]string{"https://example.com"})(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Origin", "https://example.com")
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	// Check CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "https://example.com" {
		t.Errorf("Expected CORS origin 'https://example.com', got '%s'", rr.Header().Get("Access-Control-Allow-Origin"))
	}

	if rr.Header().Get("Access-Control-Allow-Methods") == "" {
		t.Error("Missing Access-Control-Allow-Methods header")
	}

	if rr.Header().Get("Access-Control-Allow-Headers") == "" {
		t.Error("Missing Access-Control-Allow-Headers header")
	}
}

func TestCORS_Wildcard(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := CORS([]string{"*"})(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Origin", "https://anydomain.com")
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Header().Get("Access-Control-Allow-Origin") != "https://anydomain.com" {
		t.Errorf("Expected origin 'https://anydomain.com', got '%s'", rr.Header().Get("Access-Control-Allow-Origin"))
	}
}

func TestCORS_Preflight(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := CORS([]string{"*"})(handler)

	req := httptest.NewRequest("OPTIONS", "/test", nil)
	req.Header.Set("Origin", "https://example.com")
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Preflight should return 200, got %d", rr.Code)
	}
}

func TestCORS_UnauthorizedOrigin(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := CORS([]string{"https://allowed.com"})(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Origin", "https://notallowed.com")
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	// Should NOT have CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "" {
		t.Error("Unauthorized origin should not receive CORS headers")
	}
}

func TestTimeout_Success(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(10 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	})

	middleware := Timeout(1 * time.Second)(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, rr.Code)
	}
}

func TestTimeout_Exceeded(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(100 * time.Millisecond)
		w.WriteHeader(http.StatusOK)
	})

	middleware := Timeout(50 * time.Millisecond)(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	if rr.Code != http.StatusGatewayTimeout {
		t.Errorf("Expected status %d, got %d", http.StatusGatewayTimeout, rr.Code)
	}
}

func TestTimeout_ContextCancellation(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Check if context has timeout
		_, hasDeadline := r.Context().Deadline()
		if !hasDeadline {
			t.Error("Request context should have deadline")
		}
		w.WriteHeader(http.StatusOK)
	})

	middleware := Timeout(5 * time.Second)(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)
}

func TestSecurityHeaders(t *testing.T) {
	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	middleware := SecurityHeaders(handler)

	req := httptest.NewRequest("GET", "/test", nil)
	rr := httptest.NewRecorder()

	middleware.ServeHTTP(rr, req)

	// Check all security headers
	expectedHeaders := map[string]string{
		"X-Content-Type-Options":    "nosniff",
		"X-Frame-Options":           "DENY",
		"X-XSS-Protection":          "1; mode=block",
		"Strict-Transport-Security": "max-age=31536000; includeSubDomains",
		"Content-Security-Policy":   "default-src 'self'",
		"Referrer-Policy":           "no-referrer",
	}

	for header, expected := range expectedHeaders {
		if rr.Header().Get(header) != expected {
			t.Errorf("Header %s: expected '%s', got '%s'", header, expected, rr.Header().Get(header))
		}
	}
}

func TestMiddleware_Chaining(t *testing.T) {
	// Test that multiple middlewares can be chained
	limiter := NewRateLimiter(100, time.Minute)

	handler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify security headers are present
		if r.Header.Get("X-Test-Header") != "test" {
			// Headers should be set in response, not request
		}
		w.WriteHeader(http.StatusOK)
	})

	var httpHandler http.Handler = handler
	httpHandler = SecurityHeaders(httpHandler)
	httpHandler = CORS([]string{"*"})(httpHandler)
	httpHandler = RateLimit(limiter)(httpHandler)
	httpHandler = Timeout(30 * time.Second)(httpHandler)

	req := httptest.NewRequest("GET", "/test", nil)
	req.Header.Set("Origin", "https://example.com")
	req.RemoteAddr = "127.0.0.1:12345"
	rr := httptest.NewRecorder()

	httpHandler.ServeHTTP(rr, req)

	if rr.Code != http.StatusOK {
		t.Errorf("Expected status %d, got %d", http.StatusOK, rr.Code)
	}

	// Verify security headers are set
	if rr.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Error("Security headers not applied in chain")
	}

	// Verify CORS headers
	if rr.Header().Get("Access-Control-Allow-Origin") != "https://example.com" {
		t.Error("CORS headers not applied in chain")
	}
}

func TestRateLimiter_ConcurrentAccess(t *testing.T) {
	limiter := NewRateLimiter(100, time.Minute)

	// Test concurrent access
	done := make(chan bool)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 10; j++ {
				ip := "10.0.0." + strings.Repeat("1", j%3+1)
				limiter.Allow(ip)
			}
			done <- true
		}()
	}

	// Wait for all goroutines
	for i := 0; i < 10; i++ {
		<-done
	}

	// If we got here without panic, test passed
	t.Log("Concurrent access test passed")
}
