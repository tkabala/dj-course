package drivers

// ContractType represents the type of contract a driver has.
type ContractType string

const (
	Contractor ContractType = "CONTRACTOR"
	FullTime   ContractType = "FULL_TIME"
)

// Driver represents a driver entity.
type Driver struct {
	ID           int
	FirstName    string
	LastName     string
	Email        string
	Phone        string
	ContractType ContractType
}
